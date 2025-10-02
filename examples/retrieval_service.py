import json
import random
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
from datasets import load_dataset

# Load real HotpotQA data
print("Loading HotpotQA data...")
dataset = load_dataset("hotpot_qa", "distractor")
docs = []

# Extract supporting facts as documents
for item in dataset['train']:
    for fact in item['supporting_facts']:
        docs.append({
            "title": fact[0],  # Wikipedia article title
            "text": f"Article: {fact[0]}\nContent: {fact[1]}"  # Simplified
        })

print(f"Loaded {len(docs)} documents")

class RetrievalHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith('/search'):
            # Parse query parameters
            query_params = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            query = query_params.get('q', [''])[0]
            k = int(query_params.get('k', ['5'])[0])
            
            # Simple keyword matching
            query_words = set(query.lower().split())
            scored_docs = []
            
            for doc in docs:
                doc_words = set(doc['text'].lower().split())
                score = len(query_words.intersection(doc_words))
                if score > 0:
                    scored_docs.append((score, doc))
            
            # Sort by score and return top-k
            scored_docs.sort(key=lambda x: x[0], reverse=True)
            selected = [doc for _, doc in scored_docs[:k]]
            
            # If not enough matches, add random ones
            while len(selected) < k and len(selected) < len(docs):
                random_doc = random.choice(docs)
                if random_doc not in selected:
                    selected.append(random_doc)
            
            # Send response
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {"docs": selected}
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.end_headers()

if __name__ == "__main__":
    server = HTTPServer(('0.0.0.0', 8002), RetrievalHandler)  # Changed to port 8002
    print("Starting retrieval service on port 8002...")
    server.serve_forever()
