/*
    Cache-Aware Load Balancing Router

    This router combines two strategies to optimize both cache utilization and request distribution:

    1. Cache-Aware Routing (Approximate Tree)
    2. Load Balancing (Shortest Queue with Balance Thresholds)

    The router dynamically switches between these strategies based on load conditions:
    - Uses load balancing when the system is imbalanced
    - Uses cache-aware routing when the system is balanced

    A system is considered imbalanced if both conditions are met:
    1. (max - min) > abs_threshold
    2. max > rel_threshold * min

    Strategy Details:

    1. Cache-Aware Routing (Approximate Tree)
    -------------------------------------------
    This strategy maintains an approximate radix tree for each worker based on request history,
    eliminating the need for direct cache state queries. The tree stores raw text characters
    instead of token IDs to avoid tokenization overhead.

    Process:
    a. For each request, find the worker with the highest prefix match
    b. If match rate > cache_threshold:
    Route to the worker with highest match (likely has relevant data cached)
    c. If match rate ≤ cache_threshold:
    Route to the worker with smallest tree size (most available cache capacity)
    d. Background maintenance:
    Periodically evict least recently used leaf nodes to prevent memory overflow

    2. Load Balancing (Shortest Queue)
    -------------------------------------------
    This strategy tracks pending request counts per worker and routes new requests
    to the least busy worker when the system is detected to be imbalanced.

    Configuration Parameters:
    ------------------------
    1. cache_threshold: (float, 0.0 to 1.0)
    Minimum prefix match ratio to use highest-match routing.
    Below this threshold, routes to worker with most available cache space.

    2. balance_abs_threshold: (integer)
    Absolute difference threshold for load imbalance detection.
    System is potentially imbalanced if (max_load - min_load) > abs_threshold

    3. balance_rel_threshold: (float)
    Relative ratio threshold for load imbalance detection.
    System is potentially imbalanced if max_load > min_load * rel_threshold
    Used in conjunction with abs_threshold to determine final imbalance state.

    4. eviction_interval_secs: (integer)
    Interval between LRU eviction cycles for the approximate trees.

    5. max_tree_size: (integer)
    Maximum nodes per tree. When exceeded, LRU leaf nodes are evicted
    during the next eviction cycle.
*/

use super::{get_healthy_worker_indices, CacheAwareConfig, LoadBalancingPolicy, RequestHeaders};
use crate::core::Worker;
use crate::metrics::RouterMetrics;
use crate::policies::normalize_model_key;
use crate::tree::TokenTree;
use dashmap::DashMap;
use rand::Rng;
use std::collections::HashMap;
use std::sync::Arc;
use std::thread;
use std::time::Duration;
use tracing::{debug, info, warn};

/// Cache-aware routing policy
///
/// Routes requests based on cache affinity when load is balanced,
/// switches to shortest-queue routing when load is imbalanced.
/// Maintains separate trees per model for multi-model support.
#[derive(Debug)]
pub struct CacheAwarePolicy {
    config: CacheAwareConfig,
    trees: Arc<DashMap<String, Arc<TokenTree>>>, // model_id -> Arc<TokenTree>
    eviction_handle: Option<thread::JoinHandle<()>>,
    client: reqwest::Client,
}

impl CacheAwarePolicy {
    pub fn new() -> Self {
        Self::with_config(CacheAwareConfig::default())
    }

    pub fn with_config(config: CacheAwareConfig) -> Self {
        let trees = Arc::new(DashMap::<String, Arc<TokenTree>>::new());

        // Start background eviction thread if configured
        let eviction_handle = if config.eviction_interval_secs > 0 {
            let trees_clone = Arc::clone(&trees);
            let max_tree_size = config.max_tree_size;
            let interval = config.eviction_interval_secs;

            Some(thread::spawn(move || loop {
                thread::sleep(Duration::from_secs(interval));

                // Evict for all model trees
                for entry in trees_clone.iter() {
                    let model_id = entry.key();
                    let tree = entry.value();
                    tree.evict_tenant_by_size(max_tree_size);
                    debug!(
                        "Cache eviction completed for model {}, max_size: {}",
                        model_id, max_tree_size
                    );
                }
            }))
        } else {
            None
        };

        Self {
            config,
            trees,
            eviction_handle,
            client: reqwest::Client::new(),
        }
    }

    /// Add a single worker to the tree (incremental update)
    pub fn add_worker(&self, worker: &dyn Worker) {
        let tree_key = normalize_model_key(worker.model_id());
        let tree = self
            .trees
            .entry(tree_key.to_string())
            .or_insert_with(|| Arc::new(TokenTree::new()));
        tree.insert(&[], worker.url());
    }

    /// Add a worker by URL and model (for backward compatibility)
    pub fn add_worker_by_url(&self, url: &str, model_id: &str) {
        let tree = self
            .trees
            .entry(model_id.to_string())
            .or_insert_with(|| Arc::new(TokenTree::new()));
        tree.insert(&[], url);
    }

    /// Remove a worker from the tree
    pub fn remove_worker(&self, worker: &dyn Worker) {
        let tree_key = normalize_model_key(worker.model_id());
        if let Some(tree) = self.trees.get(tree_key) {
            tree.remove_tenant(worker.url());
        }
    }

    /// Remove a worker by URL (removes from all model trees for backward compatibility)
    pub fn remove_worker_by_url(&self, url: &str) {
        // Remove from all trees since we don't know which model it belongs to
        for tree_ref in self.trees.iter() {
            tree_ref.value().remove_tenant(url);
        }
    }

    /// Run cache eviction to prevent unbounded growth
    pub fn evict_cache(&self, max_size: usize) {
        for tree_ref in self.trees.iter() {
            let model_id = tree_ref.key();
            let tree = tree_ref.value();
            tree.evict_tenant_by_size(max_size);
            debug!(
                "Cache eviction for model {}, max_size: {}",
                model_id, max_size
            );
        }
    }

}

/// Score a worker for routing using the unified cache-aware formula.
///
/// Returns `None` when the worker should be hard-excluded (KV utilization at or above
/// `kv_high_watermark`). Otherwise returns a score where higher is better.
///
/// Formula:
///   token_pressure = (running_tokens / token_capacity)^2
///   kv_pressure    = kv_util^3
///   load_penalty   = alpha * token_pressure + beta * kv_pressure
///   cache_value    = cache_match * ln(1 + input_tokens)
///   score          = cache_value - lambda * load_penalty
fn score_worker(
    worker: &dyn Worker,
    cache_match: f32,
    input_tokens: usize,
    config: &CacheAwareConfig,
) -> Option<f32> {
    let kv_util = worker.kv_cache_utilization();

    if kv_util >= config.kv_high_watermark {
        return None; // hard exclusion
    }

    let token_pressure = {
        let r = worker.running_tokens() as f32 / config.token_capacity as f32;
        r * r
    };
    let kv_pressure = kv_util * kv_util * kv_util;
    let load_penalty = config.alpha * token_pressure + config.beta * kv_pressure;
    let cache_value = cache_match * (1.0_f32 + input_tokens as f32).ln();

    Some(cache_value - config.lambda * load_penalty)
}

/// Tokenize `prompt` by calling the vLLM `/tokenize` endpoint on `worker_url`.
/// Returns `(token_ids, token_count)` on success.
async fn tokenize_prompt(
    client: &reqwest::Client,
    worker_url: &str,
    model: &str,
    prompt: &str,
) -> anyhow::Result<(Vec<u32>, usize)> {
    let url = format!("{}/tokenize", worker_url.trim_end_matches('/'));
    let body = serde_json::json!({ "model": model, "prompt": prompt });

    let resp = client
        .post(&url)
        .json(&body)
        .send()
        .await?
        .error_for_status()?;

    let json: serde_json::Value = resp.json().await?;
    let tokens: Vec<u32> = json["tokens"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("missing `tokens` field"))?
        .iter()
        .map(|v| v.as_u64().unwrap_or(0) as u32)
        .collect();
    let count = tokens.len();
    Ok((tokens, count))
}

impl LoadBalancingPolicy for CacheAwarePolicy {
    fn select_worker_with_headers(
        &self,
        workers: &[Arc<dyn Worker>],
        request_text: Option<&str>,
        _headers: Option<&RequestHeaders>,
    ) -> Option<usize> {
        let healthy_indices = get_healthy_worker_indices(workers);

        if healthy_indices.is_empty() {
            return None;
        }

        // Determine the model for this set of workers (router pre-filters by model)
        // All workers should be from the same model
        let model_id = normalize_model_key(workers[healthy_indices[0]].model_id());

        // Tokenize the prompt on a randomly selected healthy worker.
        // Falls back to empty token slice when no runtime is available (unit tests)
        // or when the tokenize endpoint returns an error.
        let token_ids: Vec<u32> = if let Some(text) = request_text {
            if let Ok(handle) = tokio::runtime::Handle::try_current() {
                let worker_url = {
                    let mut rng = rand::rng();
                    let idx = rng.random_range(0..healthy_indices.len());
                    workers[healthy_indices[idx]].url().to_string()
                };
                let model = workers[healthy_indices[0]].model_id().to_string();
                let text_owned = text.to_string();
                let client = self.client.clone();

                match tokio::task::block_in_place(|| {
                    handle.block_on(tokenize_prompt(&client, &worker_url, &model, &text_owned))
                }) {
                    Ok((ids, _)) => ids,
                    Err(e) => {
                        warn!("Tokenize failed for model '{}': {}", model_id, e);
                        vec![]
                    }
                }
            } else {
                // No tokio runtime (e.g. unit tests) — skip tokenization
                vec![]
            }
        } else {
            vec![]
        };

        let input_tokens = token_ids.len();

        // Retrieve tree for this model (may be absent for newly-initialized models)
        let tree = self.trees.get(model_id).map(|entry| entry.value().clone());

        if tracing::enabled!(tracing::Level::DEBUG) {
            let keys: Vec<_> = self.trees.iter().map(|entry| entry.key().clone()).collect();
            debug!("Available tree keys: {:?}", keys);
        }

        // Compute prefix-match result; use a zeroed sentinel when no tree exists yet
        let result = match &tree {
            Some(t) => t.prefix_match_with_counts(&token_ids),
            None => {
                debug!(
                    "No tree found for model '{}', proceeding with zero cache match",
                    model_id
                );
                // Safe zero-valued result: all workers get cache_match = 0.0
                crate::tree::PrefixMatchTokenResult {
                    tenant: String::new().into(),
                    matched_token_count: 0,
                    input_token_count: input_tokens,
                }
            }
        };

        debug!(
            "Cache match for model '{}': matched_tokens={}, input_tokens={}, best_tenant='{}'",
            model_id, result.matched_token_count, result.input_token_count, result.tenant
        );

        // Score every healthy worker using the unified formula.
        // cache_match is non-zero only for the worker identified by the prefix tree.
        let tenant_url: &str = &result.tenant;
        let mut best_idx: Option<usize> = None;
        let mut best_score = f32::NEG_INFINITY;

        for &idx in &healthy_indices {
            let worker = &*workers[idx];
            let cache_match = if !tenant_url.is_empty() && worker.url() == tenant_url {
                result.matched_token_count as f32 / result.input_token_count.max(1) as f32
            } else {
                0.0_f32
            };

            if let Some(score) = score_worker(worker, cache_match, input_tokens, &self.config) {
                debug!(
                    "Worker '{}': cache_match={:.3}, score={:.4}",
                    worker.url(),
                    cache_match,
                    score
                );
                if score > best_score {
                    best_score = score;
                    best_idx = Some(idx);
                }
            } else {
                debug!(
                    "Worker '{}' hard-excluded (kv_util={:.3} >= watermark={:.3})",
                    worker.url(),
                    worker.kv_cache_utilization(),
                    self.config.kv_high_watermark
                );
            }
        }

        // Fallback when ALL workers are hard-excluded: pick lowest KV utilization
        let selected_idx = if best_idx.is_none() {
            warn!(
                "All healthy workers for model '{}' are hard-excluded by kv_high_watermark; \
                 falling back to lowest-KV worker",
                model_id
            );
            healthy_indices
                .iter()
                .min_by(|&&a, &&b| {
                    workers[a]
                        .kv_cache_utilization()
                        .partial_cmp(&workers[b].kv_cache_utilization())
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .copied()
        } else {
            best_idx
        };

        let Some(idx) = selected_idx else {
            return None;
        };

        // Insert token_ids into tree for the selected worker
        if !token_ids.is_empty() {
            if let Some(t) = &tree {
                t.insert(&token_ids, workers[idx].url());
            }
        }

        workers[idx].increment_processed();
        RouterMetrics::record_processed_request(workers[idx].url());
        RouterMetrics::record_policy_decision(self.name(), workers[idx].url());

        Some(idx)
    }

    fn name(&self) -> &'static str {
        "cache_aware"
    }

    fn needs_request_text(&self) -> bool {
        true // Cache-aware policy needs request text for cache affinity
    }

    fn on_request_complete(&self, worker_url: &str, success: bool, tokens: usize) {
        // Token decrement is handled by the caller (router) via decrement_load_with_tokens.
        // Log completion details for observability.
        tracing::debug!(
            "Request to {} completed: success={}, tokens={}",
            worker_url,
            success,
            tokens
        );
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn select_worker_pair_with_headers(
        &self,
        prefill_workers: &[Arc<dyn Worker>],
        decode_workers: &[Arc<dyn Worker>],
        request_text: Option<&str>,
        headers: Option<&RequestHeaders>,
    ) -> Option<(usize, usize)> {
        // DEPRECATED: This method is no longer used when separate policies are configured.
        // The PD router now uses separate policies for prefill and decode selection.
        // This implementation remains for backward compatibility when a single policy is used.

        // In PD mode with single policy:
        // - Prefill: Use cache-aware routing for better cache utilization
        // - Decode: Use least-load routing for better load distribution

        // Select prefill worker using cache-aware logic
        let prefill_idx =
            self.select_worker_with_headers(prefill_workers, request_text, headers)?;

        // Select decode worker using least-load logic
        let healthy_decode = get_healthy_worker_indices(decode_workers);
        if healthy_decode.is_empty() {
            return None;
        }

        let decode_idx = healthy_decode
            .iter()
            .min_by_key(|&&idx| decode_workers[idx].load())
            .copied()?;

        Some((prefill_idx, decode_idx))
    }

    fn requires_initialization(&self) -> bool {
        true // Cache-aware policy requires init_workers() to set up trees
    }

    fn init_workers(&self, workers: &[Arc<dyn Worker>]) {
        // Group workers by model
        info!(
            "Initializing workers for cache-aware policy: {}",
            workers
                .iter()
                .map(|w| w.url())
                .collect::<Vec<_>>()
                .join(", ")
        );
        let mut model_workers: HashMap<String, Vec<&Arc<dyn Worker>>> = HashMap::new();
        for worker in workers {
            let tree_key = normalize_model_key(worker.model_id());
            model_workers
                .entry(tree_key.to_string())
                .or_default()
                .push(worker);
        }

        // Initialize tree for each model
        for (tree_key, model_workers) in model_workers {
            info!(
                "Creating tree for model key: '{}' with {} workers",
                tree_key,
                model_workers.len()
            );
            let tree = self
                .trees
                .entry(tree_key)
                .or_insert_with(|| Arc::new(TokenTree::new()))
                .clone();
            for worker in model_workers {
                tree.insert(&[], worker.url());
            }
        }
    }
}

impl Default for CacheAwarePolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for CacheAwarePolicy {
    fn drop(&mut self) {
        // Note: We can't properly stop the eviction thread since it's in an infinite loop
        // In a production system, we'd use a channel or atomic flag to signal shutdown
        if let Some(handle) = self.eviction_handle.take() {
            // The thread will continue running until the program exits
            // This is acceptable for now since the router typically runs for the lifetime of the program
            drop(handle);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BasicWorker, WorkerType};

    #[test]
    fn test_cache_aware_with_balanced_load() {
        // Create policy without eviction thread for testing
        let config = CacheAwareConfig {
            eviction_interval_secs: 0, // Disable eviction thread
            ..Default::default()
        };
        let policy = CacheAwarePolicy::with_config(config);
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(BasicWorker::new(
                "http://w1:8000".to_string(),
                WorkerType::Regular,
            )),
            Arc::new(BasicWorker::new(
                "http://w2:8000".to_string(),
                WorkerType::Regular,
            )),
        ];

        // Initialize the policy with workers
        policy.init_workers(&workers);

        // First request should be distributed
        let idx1 = policy.select_worker(&workers, Some("hello world")).unwrap();

        // Same request should go to same worker (cache hit)
        let idx2 = policy.select_worker(&workers, Some("hello world")).unwrap();
        assert_eq!(idx1, idx2);

        // Similar request should also go to same worker
        let idx3 = policy.select_worker(&workers, Some("hello")).unwrap();
        assert_eq!(idx1, idx3);
    }

    #[test]
    fn test_cache_aware_with_imbalanced_load() {
        let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
            cache_threshold: 0.5,
            balance_abs_threshold: 5,
            balance_rel_threshold: 2.0,
            eviction_interval_secs: 0, // Disable eviction thread
            max_tree_size: 10000,
            ..Default::default()
        });

        let worker1 = BasicWorker::new("http://w1:8000".to_string(), WorkerType::Regular);
        let worker2 = BasicWorker::new("http://w2:8000".to_string(), WorkerType::Regular);

        // Create significant load imbalance
        for _ in 0..20 {
            worker1.increment_load();
        }
        // worker2 has load 0

        let workers: Vec<Arc<dyn Worker>> = vec![Arc::new(worker1), Arc::new(worker2)];
        policy.init_workers(&workers);

        // Should select worker2 (lower load) despite cache affinity
        for _ in 0..5 {
            let idx = policy.select_worker(&workers, Some("test")).unwrap();
            assert_eq!(idx, 1); // Should always pick worker2
        }
    }

    #[test]
    fn test_cache_aware_worker_removal() {
        let config = CacheAwareConfig {
            eviction_interval_secs: 0, // Disable eviction thread
            ..Default::default()
        };
        let policy = CacheAwarePolicy::with_config(config);
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(BasicWorker::new(
                "http://w1:8000".to_string(),
                WorkerType::Regular,
            )),
            Arc::new(BasicWorker::new(
                "http://w2:8000".to_string(),
                WorkerType::Regular,
            )),
        ];

        policy.init_workers(&workers);

        // Route some requests
        policy.select_worker(&workers, Some("test1"));
        policy.select_worker(&workers, Some("test2"));

        // Remove a worker
        policy.remove_worker_by_url("http://w1:8000");
        workers[0].set_healthy(false);

        // All requests should now go to worker2
        let idx = policy.select_worker(&workers, Some("test1")).unwrap();
        assert_eq!(idx, 1);
    }
}
