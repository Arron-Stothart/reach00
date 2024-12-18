from typing import List, Dict
import torch

class PatternSeparator:
    """Basic pattern separation mechanism
    
    The CA3 region of the hippocampus helps determine if an input pattern is similar enough to trigger completion 
    (treating events as the same) or different enough to trigger separation (treating them as distinct).
    """
    
    def __init__(self, 
                 distinctiveness_threshold: float = 0.5,
                 feature_weights: Dict[str, float] = None):
        """
        Args:
            distinctiveness_threshold: Minimum difference required between episodes
            feature_weights: Weights for different features (entities, actions, locations)
        """
        self.distinctiveness_threshold = distinctiveness_threshold # TODO: Dynamic threshold
        self.feature_weights = feature_weights or {
            'entities': 1.0,
            'actions': 0.8,
            'locations': 0.6
        }
    
    def calculate_distinctiveness(self, 
                                episode1: torch.Tensor, 
                                episode2: torch.Tensor) -> float:
        """
        Calculate distinctiveness score between two episodes
        Higher score = more distinct
        
        Memory for LLMs is text-only, we can use:
        - Semantic similarity of core elements (actors, actions, etc.)
        - Semantic similarity of contextual elements / circumstantial details (location, time, etc.)
        - Abstract relationships
        - Temporal context (when events occurred in the episode sequence)
        """
        # Basic cosine similarity
        similarity = torch.nn.functional.cosine_similarity(
            episode1.unsqueeze(0), 
            episode2.unsqueeze(0)
        )
        return 1.0 - similarity.item()
    
    def enhance_separation(self, 
                          episodes: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Enhance separation between similar episodes by amplifying differences
        """
        enhanced_episodes = []
        
        for i, current_episode in enumerate(episodes):
            # Compare with previous episodes
            needs_separation = False
            
            for prev_episode in enhanced_episodes:
                distinctiveness = self.calculate_distinctiveness(
                    current_episode, prev_episode
                )
                
                if distinctiveness < self.distinctiveness_threshold:
                    needs_separation = True
                    break
            
            if needs_separation:
                # Enhance distinctive features
                enhanced_episode = self._amplify_differences(
                    current_episode, 
                    enhanced_episodes
                )
            else:
                enhanced_episode = current_episode
                
            enhanced_episodes.append(enhanced_episode)
            
        return enhanced_episodes
    
    def _amplify_differences(self, 
                           episode: torch.Tensor,
                           other_episodes: List[torch.Tensor]) -> torch.Tensor:
        """
        Amplify features that make this episode distinct from others
        """
        # Start with a copy of the episode
        enhanced = episode.clone()
        
        # Calculate average of other episodes
        if other_episodes:
            others_mean = torch.stack(other_episodes).mean(dim=0)
            
            # Find most distinctive dimensions
            differences = torch.abs(episode - others_mean)
            
            # Amplify top distinctive features
            top_k = int(0.2 * differences.shape[0])  # Amplify top 20% most distinctive features
            _, indices = torch.topk(differences, top_k)
            
            # Increase the difference in these dimensions
            enhancement_factor = 1.2  # Increase distinctiveness by 20%
            enhanced[indices] += (episode[indices] - others_mean[indices]) * (enhancement_factor - 1.0)
            
        return enhanced
