import numpy as np
from datetime import datetime

class AnalyticsDashboard:
    def __init__(self):
        self.search_history = []

    def log_search(self, query_type, query, num_results, avg_similarity):
        self.search_history.append({
            'timestamp': datetime.now().isoformat(),
            'query_type': query_type,
            'query': str(query)[:100],
            'num_results': num_results,
            'avg_similarity': avg_similarity
        })

    def get_stats(self):
        if not self.search_history:
            return "No searches yet"

        total = len(self.search_history)
        text_searches = sum(1 for s in self.search_history if s['query_type'] == 'text')
        image_searches = total - text_searches
        avg_sim = np.mean([s['avg_similarity'] for s in self.search_history])

        return f"""**Search Analytics**
- Total Searches: {total}
- Text Searches: {text_searches}
- Image Searches: {image_searches}
- Average Similarity: {avg_sim:.4f}
- Last Search: {self.search_history[-1]['timestamp']}"""
