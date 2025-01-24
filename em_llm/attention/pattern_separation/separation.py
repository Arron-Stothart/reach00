import torch
from typing import List

class EnhancedPatternSeparator():
    """
    Pattern separation for transformer memory blocks
    
    The CA3 region of the hippocampus helps determine if an input pattern is similar enough to trigger completion 
    (treating events as the same) or different enough to trigger separation (treating them as distinct).
    
    TODO: Use more sophisticated similarity metrics
        - Semantic similarity of core elements (actors, actions, etc.)
        - Semantic similarity of contextual elements / circumstantial details (location, time, etc.)
        - Abstract relationships
        - Temporal context (when events occurred in the episode sequence)
        - Hierarchical pattern separation (Refine on high-level features, then refine on lower-level features)
        - Some sort of confidence metric
    """
    
    def __init__(self,
                distinctiveness_threshold: float = 0.5,
                enhancement_factor: float = 1.2,
                retrieval_threshold: float = 0.7,
                max_retrievals: int = 20,
                # feature_weights: Dict[str, float] = None
        ):
        self.distinctiveness_threshold = distinctiveness_threshold
        self.enhancement_factor = enhancement_factor
        self.retrieval_threshold = retrieval_threshold
        self.max_retrievals = max_retrievals
        # self.feature_weights = feature_weights

    def enhance_separation(self, current: torch.Tensor, block_repr_k: List[torch.Tensor]) -> torch.Tensor:
        """Enhanced pattern separation with memory retrieval"""
        # 1. Retrieve similar memories using block_repr_k
        similarities = []
        for block in block_repr_k:
            sim = self.calculate_distinctiveness(current, block)
            similarities.append(sim)
        
        # 2. Limit number of retrievals
        similar_indices = [i for i, sim in enumerate(similarities) 
                         if sim > self.retrieval_threshold]
        
        if len(similar_indices) > self.max_retrievals:
            similar_indices = sorted(similar_indices, 
                                  key=lambda i: similarities[i],
                                  reverse=True)[:self.max_retrievals]
        if not similar_indices:
            return current

        # 3. Calculate mean representation of similar blocks
        similar_blocks = [block_repr_k[i] for i in similar_indices]
        similar_mean = torch.stack(similar_blocks).mean(dim=0)
        
        # 4. If similar to mean representation of similar blocks, enhance separation
        # TODO: Case for when blocks come from same episode
        adaptive_threshold = self._calculate_adaptive_threshold(similarities)
        if self.calculate_distinctiveness(current, similar_mean) < adaptive_threshold:
            diff = current - similar_mean
            enhanced = current + (diff * (self.enhancement_factor - 1.0))
            enhanced = enhanced * (torch.norm(current) / torch.norm(enhanced)) # Normalize
            
            return enhanced
            
        return current
    
    def _calculate_adaptive_threshold(self, similarities: List[float]) -> float:
        """Adjusts threshold based on memory density"""
        if not similarities:
            return self.distinctiveness_threshold
        
        density = sum(1 for s in similarities if s > self.retrieval_threshold) / len(similarities)
        return self.distinctiveness_threshold * (1 + density * 0.5)

    def _amplify_differences(self, 
                           episode: torch.Tensor,
                           similar_episodes: List[torch.Tensor],
                           similarities: List[float]) -> torch.Tensor:
        """
        Amplify differences weighted by similarity
        Finds dimensions with largest differences and amplifies them
        """
        if not similar_episodes:
            return episode
            
        weights = torch.softmax(torch.tensor(similarities), dim=0)
        weighted_episodes = torch.stack([ep * w for ep, w in zip(similar_episodes, weights)])
        weighted_mean = weighted_episodes.mean(dim=0)
        
        diff_vector = episode - weighted_mean
        diff_magnitude = torch.abs(diff_vector)
        
        k = int(0.2 * diff_vector.shape[-1])
        top_diffs, indices = torch.topk(diff_magnitude, k)
        
        enhanced = episode.clone()
        
        enhanced[indices] += diff_vector[indices] * (self.enhancement_factor - 1.0)
        
        enhanced = enhanced * (torch.norm(episode) / torch.norm(enhanced)) # Normalize
        
        return enhanced