from flask import Flask, request, jsonify, send_from_directory
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
import os, json, tempfile
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__, static_folder='static')

# ── Sample B2B knowledge base ──────────────────────────────────────────────
KB = """
IndiaMART is India's largest B2B marketplace connecting buyers and suppliers.
Suppliers can list products across categories like machinery, chemicals, textiles, electronics, and agriculture.
Buyers can search for suppliers using filters like location, price range, and minimum order quantity.
IndiaMART has over 87 lakh responsive suppliers and 129 million product listings.
The platform supports SMEs, large enterprises, and individual sellers.
Registered buyers can send inquiries directly to suppliers via the platform.
IndiaMART's mobile app has over 10 crore downloads on Google Play Store.
Payment Protection Plan (PPP) ensures safe transactions between buyers and suppliers.
Trade Alerts notify buyers when new matching suppliers join the platform.
Verified suppliers have their business credentials checked by IndiaMART.
Bulk inquiry feature allows buyers to contact multiple suppliers simultaneously.
IndiaMART offers premium membership plans for enhanced visibility and lead generation.
TrustSEAL is IndiaMART's trust badge awarded to verified and active suppliers.
Categories on IndiaMART include industrial machinery, building materials, IT hardware, food products, and fashion.
IndiaMART was founded in 1999 and is headquartered in Noida, India.
The company went public with its IPO in 2019 on NSE and BSE.
IndiaMART's mission is to make doing business easy for buyers and suppliers across India.
Suppliers can track buyer inquiries and respond to them through the supplier dashboard.
The platform uses AI-powered lead matching to connect the right buyers with relevant suppliers.
IndiaMART's Imprego product helps suppliers manage their business operations digitally.
"""

# ── Build vector store on startup ─────────────────────────────────────────
_kb_file = os.path.join(tempfile.gettempdir(), "kb.txt")
with open(_kb_file, "w") as f:
    f.write(KB)

loader = TextLoader(_kb_file)
docs   = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)
chunks = splitter.split_documents(docs)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever   = vectorstore.as_retriever(search_kwargs={"k": 3})

_qa_chain = None

def get_qa_chain():
    global _qa_chain
    if _qa_chain is not None:
        return _qa_chain
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        return None
    llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key, temperature=0.2)
    _qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=False)
    return _qa_chain

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data     = request.json
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "No question provided"}), 400
    chain = get_qa_chain()
    if chain is None:
        return jsonify({"error": "GROQ_API_KEY is not set. Set it as an environment variable and restart the server."}), 503
    result = chain.invoke({"query": question})
    return jsonify({"answer": result["result"]})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
