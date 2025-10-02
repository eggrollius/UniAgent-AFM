import json
import random
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler

# Simple document collection
docs = [
    {"title": "France", "text": "France is a country in Europe. The capital of France is Paris."},
    {"title": "Paris", "text": "Paris is the capital and largest city of France."},
    {"title": "Europe", "text": "Europe is a continent. France is located in Western Europe."},
    {"title": "Capital Cities", "text": "Capital cities are the main cities of countries. Paris is the capital of France."},
    {"title": "Geography", "text": "France is located in Western Europe. Its capital is Paris."}
]

class RetrievalHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith('/search'):
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
    server = HTTPServer(('0.0.0.0', 8002), RetrievalHandler)
    print("Starting quick retrieval service on port 8002...")
    server.serve_forever()
