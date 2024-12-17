from typing import List, Dict
import torch
import json
from transformers import AutoTokenizer, MistralForCausalLM
from .separation import PatternSeparator

# NOTE: LLM-GENERATED TEST SEQUENCES - EARLY TESTING
TEST_SEQUENCES = {
    "similar_actions": [
        """John walked into the coffee shop on Main Street. He ordered a large latte 
        and sat by the window. The barista called his name after a few minutes.""",
        
        """Sarah walked into the coffee shop on Oak Street. She ordered a small cappuccino 
        and sat by the counter. The barista called her name after a few minutes.""",
        
        """Mike walked into the coffee shop on Pine Street. He ordered a medium americano 
        and sat by the door. The barista called his name after a few minutes."""
    ],
    
    "similar_locations": [
        """The museum's ancient Egyptian exhibit featured a golden sarcophagus. 
        Tourists gathered around, taking photos of the intricate hieroglyphics.""",
        
        """The museum's Greek exhibit displayed marble statues of gods. 
        Students sketched in their notebooks, studying the classical forms.""",
        
        """The museum's Roman exhibit showcased ancient pottery and coins. 
        Researchers examined the artifacts, documenting their findings."""
    ],
    
    "similar_events": [
        """During the graduation ceremony, Emma gave the valedictorian speech. 
        Her parents watched proudly as she received her diploma with honors.""",
        
        """During the graduation ceremony, James performed a musical piece. 
        His family cheered loudly as he accepted his diploma with distinction.""",
        
        """During the graduation ceremony, Lisa led the pledge of allegiance. 
        Her grandparents smiled as she collected her diploma with merit."""
    ],
    
    "temporal_sequence": [
        """Monday morning started with a team meeting. The project timeline was discussed.
        New assignments were distributed. Progress reports were reviewed.""",
        
        """Tuesday morning began with client calls. Contract terms were negotiated.
        Proposals were updated. Deadlines were confirmed.""",
        
        """Wednesday morning opened with budget planning. Financial reports were analyzed.
        Resources were allocated. Forecasts were updated."""
    ],
    
    "mixed_similarity": [
        """The chef prepared a French dish in the kitchen. Steam rose from the pots
        as he carefully added herbs. The sous chef assisted with plating.""",
        
        """The artist painted a landscape in the studio. Light streamed through windows
        as she carefully mixed colors. The apprentice prepared canvases.""",
        
        """The scientist conducted experiments in the lab. Vapor rose from beakers
        as she carefully measured compounds. The assistant recorded data."""
    ]
}

class SeparationTester:
    """Utility class for testing pattern separation mechanisms"""
    
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model
        self.results = {}
    
    def encode_sequences(self, sequences: List[str]) -> List[torch.Tensor]:
        """Encode text sequences into tensor representations"""
        encoded = []
        for seq in sequences:
            tokens = self.tokenizer(seq, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(**tokens)
                sequence_repr = outputs.last_hidden_state.mean(dim=1)
                encoded.append(sequence_repr.squeeze())
        return encoded
    
    def test_separation(self, 
                       separator,
                       test_case: str = None) -> Dict:
        """Test pattern separation on specific or all test cases"""
        results = {}
        
        sequences = TEST_SEQUENCES[test_case] if test_case else TEST_SEQUENCES
        
        for case_name, seqs in sequences.items():
            encoded_seqs = self.encode_sequences(seqs)
            
            baseline_sims = []
            for i in range(len(encoded_seqs)):
                for j in range(i + 1, len(encoded_seqs)):
                    sim = separator.calculate_distinctiveness(
                        encoded_seqs[i],
                        encoded_seqs[j]
                    )
                    baseline_sims.append(sim)
            
            enhanced_seqs = separator.enhance_separation(encoded_seqs)
            
            enhanced_sims = []
            for i in range(len(enhanced_seqs)):
                for j in range(i + 1, len(enhanced_seqs)):
                    sim = separator.calculate_distinctiveness(
                        enhanced_seqs[i],
                        enhanced_seqs[j]
                    )
                    enhanced_sims.append(sim)
            
            results[case_name] = {
                "baseline_similarities": baseline_sims,
                "enhanced_similarities": enhanced_sims,
                "improvement": sum(enhanced_sims) - sum(baseline_sims)
            }
            
        self.results.update(results)
        return results
    
    def analyze_results(self) -> Dict:
        """Analyze separation performance across test cases"""
        analysis = {}
        
        for case_name, result in self.results.items():
            analysis[case_name] = {
                "avg_baseline_sim": sum(result["baseline_similarities"]) / len(result["baseline_similarities"]),
                "avg_enhanced_sim": sum(result["enhanced_similarities"]) / len(result["enhanced_similarities"]),
                "max_improvement": max(result["enhanced_similarities"]) - min(result["baseline_similarities"]),
                "overall_improvement": result["improvement"]
            }
            
        return analysis
    
    def print_results(self):
        """Print formatted test results"""
        analysis = self.analyze_results()
        
        print("\nPattern Separation Test Results:")
        print("=" * 50)
        
        for case_name, metrics in analysis.items():
            print(f"\nTest Case: {case_name}")
            print("-" * 30)
            print(f"Average Baseline Similarity: {metrics['avg_baseline_sim']:.3f}")
            print(f"Average Enhanced Similarity: {metrics['avg_enhanced_sim']:.3f}")
            print(f"Maximum Improvement: {metrics['max_improvement']:.3f}")
            print(f"Overall Improvement: {metrics['overall_improvement']:.3f}")

def main():
    model = MistralForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-v0.1",
        torch_dtype="auto",
        device_map="cuda"
    )
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    
    separator = PatternSeparator(
        distinctiveness_threshold=0.3,
        feature_weights={
            'entities': 1.0,
            'actions': 0.8,
            'locations': 0.6
        }
    )
    
    tester = SeparationTester(tokenizer, model)
    
    print("Testing pattern separation...")
    for case in ["similar_actions", "similar_locations", "similar_events"]:
        print(f"\nTesting {case}...")
        results = tester.test_separation(separator, case)
        print(json.dumps(results, indent=4))
    
    tester.print_results()

if __name__ == "__main__":
    main() 