"""
PageIndex RAG Evaluator — Single-file Django application
Run:  python app.py runserver
"""
import sys, os, django, json, tempfile, time, re
from django.conf import settings
from django.http import HttpResponse
from django.urls import path
from django.views.decorators.csrf import csrf_exempt

# ─── Django Settings ────────────────────────────────────────────────────────
if not settings.configured:
    settings.configure(
        DEBUG=True,
        SECRET_KEY="pageindex-rag-evaluator-dev-key-change-in-production",
        ROOT_URLCONF=__name__,
        ALLOWED_HOSTS=["*"],
    )
    django.setup()

# ─── API Clients ────────────────────────────────────────────────────────────
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pageindex import PageIndexClient
import pageindex.utils as pi_utils

load_dotenv()
try:
    pi_client = PageIndexClient(api_key=os.getenv("PAGEINDEX_API_KEY", ""))
    genai_client = genai.Client()
except Exception as e:
    print(f"Warning: Checking API keys for PageIndex & Gemini failed: {e}")
    pi_client = None
    genai_client = None

document_cache = {}

# ─── View ───────────────────────────────────────────────────────────────────
def index(request):
    return HttpResponse(HTML_TEMPLATE, content_type="text/html")

@csrf_exempt
def upload_document(request):
    if request.method == "POST" and request.FILES.get("file"):
        uploaded_file = request.FILES["file"]
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            for chunk in uploaded_file.chunks():
                temp_file.write(chunk)
            temp_path = temp_file.name
        try:
            # Submit to PageIndex
            doc_id = pi_client.submit_document(temp_path)["doc_id"]
            return HttpResponse(json.dumps({"doc_id": doc_id}), content_type="application/json")
        except Exception as e:
            return HttpResponse(json.dumps({"error": str(e)}), status=500, content_type="application/json")
    return HttpResponse(json.dumps({"error": "Invalid request"}), status=400, content_type="application/json")

def check_status(request):
    doc_id = request.GET.get("doc_id")
    if not doc_id:
        return HttpResponse(json.dumps({"error": "Missing doc_id"}), status=400)
    try:
        is_ready = pi_client.is_retrieval_ready(doc_id)
        if is_ready:
            tree = pi_client.get_tree(doc_id, node_summary=True)['result']
            document_cache[doc_id] = {
                "tree": tree,
                "node_map": pi_utils.create_node_mapping(tree)
            }
            return HttpResponse(json.dumps({"status": "ready"}), content_type="application/json")
        return HttpResponse(json.dumps({"status": "processing"}), content_type="application/json")
    except Exception as e:
        return HttpResponse(json.dumps({"error": str(e)}), status=500, content_type="application/json")

def call_llm_sync(prompt, model="gemini-2.5-flash", temperature=0):
    response = genai_client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=temperature)
    )
    text = response.text.strip()
    text = re.sub(r'^```(?:json)?\s*\n', '', text)
    text = re.sub(r'\n```\s*$', '', text)
    return text

@csrf_exempt
def query_document(request):
    if request.method == "POST":
        data = json.loads(request.body)
        query = data.get("query")
        doc_id = data.get("doc_id")
        
        if not query or not doc_id or doc_id not in document_cache:
            return HttpResponse(json.dumps({"error": "Invalid query or document not ready"}), status=400)
            
        tree = document_cache[doc_id]["tree"]
        node_map = document_cache[doc_id]["node_map"]
        
        tree_without_text = pi_utils.remove_fields(tree.copy(), fields=['text'])
        search_prompt = f"""
You are given a question and a tree structure of a document.
Each node contains a node id, node title, and a corresponding summary.
Your task is to find all nodes that are likely to contain the answer to the question.

Question: {query}

Document tree structure:
{json.dumps(tree_without_text, indent=2)}

Please reply in the following JSON format:
{{
    "thinking": "<Your thinking process on which nodes are relevant to the question>",
    "node_list": ["node_id_1", "node_id_2", ..., "node_id_n"]
}}
Directly return the final JSON structure. Do not output anything else.
"""
        try:
            tree_search_result_str = call_llm_sync(search_prompt)
            try:
                search_result = json.loads(tree_search_result_str)
            except json.JSONDecodeError:
                json_match = re.search(r'\{.*\}', tree_search_result_str, re.DOTALL)
                if json_match:
                    search_result = json.loads(json_match.group())
                else:
                    raise ValueError("Could not extract JSON")
            
            node_list = search_result.get("node_list", [])
            relevant_nodes = [node_map[nid] for nid in node_list if nid in node_map]
            relevant_content = "\n\n".join(n["text"] for n in relevant_nodes)
            
            answer_prompt = f"Answer the question based on the context:\n\nQuestion: {query}\nContext: {relevant_content}\n\nProvide a clear, concise answer based only on the context provided."
            answer = call_llm_sync(answer_prompt)
            
            frontend_chunks = []
            for i, n in enumerate(relevant_nodes):
                frontend_chunks.append({
                    "rank": i + 1,
                    "page": n.get("page_index", "N/A"),
                    "text": n.get("text", "")[:400] + "..."
                })
                
            return HttpResponse(json.dumps({
                "answer": answer,
                "thinking": search_result.get("thinking", ""),
                "chunks": frontend_chunks
            }), content_type="application/json")
        except Exception as e:
            return HttpResponse(json.dumps({"error": str(e)}), status=500, content_type="application/json")
    return HttpResponse(json.dumps({"error": "Invalid request"}), status=400)

# ─── URLs ───────────────────────────────────────────────────────────────────
urlpatterns = [
    path("", index),
    path("api/upload", upload_document),
    path("api/status", check_status),
    path("api/query", query_document),
]

# ─── HTML Template ──────────────────────────────────────────────────────────
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>PageIndex RAG Evaluator — Legal Document Retrieval Testing</title>
  <meta name="description" content="Test retrieval accuracy of your legal document RAG system with ground-truth comparisons, batch evaluation, and per-query diagnostics." />

  <!-- Tailwind CSS CDN -->
  <script src="https://cdn.tailwindcss.com"></script>
  <script>
    tailwind.config = {
      theme: {
        extend: {
          colors: {
            base:    '#FFFFFF',
            card:    '#F9FAFB',
            card2:   '#F3F4F6',
            accent:  '#6C63FF',
            accent2: '#5A52D5',
            correct: '#16A34A',
            wrong:   '#DC2626',
            near:    '#D97706',
            muted:   '#6B7280',
            border:  '#E5E7EB',
          },
          fontFamily: {
            sans: ['Inter', 'system-ui', 'sans-serif'],
            mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
          },
        },
      },
    };
  </script>

  <!-- Google Fonts -->
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet" />

  <style>
    /* ── Base ─────────────────────────────────────────── */
    * { box-sizing: border-box; }
    body { background: #FFFFFF; color: #1F2937; font-family: 'Inter', system-ui, sans-serif; }
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #FFFFFF; }
    ::-webkit-scrollbar-thumb { background: #E5E7EB; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #D1D5DB; }

    /* ── Animations ──────────────────────────────────── */
    @keyframes fadeInUp {
      from { opacity: 0; transform: translateY(12px); }
      to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes slideIn {
      from { opacity: 0; transform: translateX(-8px); }
      to   { opacity: 1; transform: translateX(0); }
    }
    @keyframes pulse-glow {
      0%, 100% { box-shadow: 0 0 0 0 rgba(108, 99, 255, 0.4); }
      50%      { box-shadow: 0 0 0 8px rgba(108, 99, 255, 0); }
    }
    @keyframes progress-stripe {
      0%   { background-position: 1rem 0; }
      100% { background-position: 0 0; }
    }
    .animate-fade-in-up { animation: fadeInUp 0.35s ease-out both; }
    .animate-slide-in   { animation: slideIn 0.25s ease-out both; }
    .pulse-glow         { animation: pulse-glow 2s ease-in-out infinite; }

    /* ── Drag & Drop ─────────────────────────────────── */
    .drop-zone {
      border: 2px dashed #E5E7EB;
      transition: all 0.25s ease;
    }
    .drop-zone.dragover {
      border-color: #6C63FF;
      background: rgba(108, 99, 255, 0.04);
    }

    /* ── Progress Bar ────────────────────────────────── */
    .progress-bar-striped {
      background-image: linear-gradient(
        45deg,
        rgba(255,255,255,.1) 25%, transparent 25%,
        transparent 50%, rgba(255,255,255,.1) 50%,
        rgba(255,255,255,.1) 75%, transparent 75%,
        transparent
      );
      background-size: 1rem 1rem;
      animation: progress-stripe 0.6s linear infinite;
    }

    /* ── Toast ────────────────────────────────────────── */
    .toast {
      position: fixed; bottom: 1.5rem; right: 1.5rem;
      padding: 0.75rem 1.25rem; border-radius: 0.5rem;
      font-weight: 500; z-index: 100;
      transform: translateY(120%); opacity: 0;
      transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .toast.show { transform: translateY(0); opacity: 1; }

    /* ── Query Card hover ────────────────────────────── */
    .query-card {
      transition: all 0.2s ease;
      border-left: 3px solid transparent;
    }
    .query-card:hover { background: #22252F; }
    .query-card.active {
      background: #22252F;
      border-left-color: #6C63FF;
    }

    /* ── Collapsible ─────────────────────────────────── */
    .collapsible-content {
      max-height: 0; overflow: hidden;
      transition: max-height 0.3s ease;
    }
    .collapsible-content.open { max-height: 500px; }

    /* ── Result Cards ────────────────────────────────── */
    .result-card {
      transition: all 0.2s ease;
    }
    .result-card:hover {
      transform: translateY(-2px);
      box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
  </style>
</head>
<body class="h-screen overflow-hidden">

  <!-- ═══════════════ LAYOUT ═══════════════ -->
  <div class="flex h-screen">

    <!-- ─── LEFT SIDEBAR (30%) ──────────────────────── -->
    <aside id="sidebar" class="w-[30%] min-w-[320px] border-r border-border flex flex-col bg-base overflow-hidden">

      <!-- Header -->
      <div class="px-5 py-4 border-b border-border flex items-center gap-3">
        <div class="w-8 h-8 rounded-lg bg-accent/10 border border-accent/20 flex items-center justify-center">
          <svg class="w-4 h-4 text-accent" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
              d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
        </div>
        <div>
          <h1 class="text-sm font-semibold text-gray-900 tracking-wide">PageIndex RAG</h1>
          <p class="text-[11px] text-muted">Evaluator</p>
        </div>
      </div>

      <!-- Scrollable content -->
      <div class="flex-1 overflow-y-auto px-4 py-4 space-y-5">

        <!-- Drop Zone -->
        <div>
          <label class="text-[11px] font-medium text-muted uppercase tracking-wider mb-2 block">Document</label>
          <div id="dropZone" class="drop-zone rounded-lg p-5 text-center cursor-pointer group">
            <div id="dropContent">
              <svg class="w-8 h-8 mx-auto text-muted mb-2 group-hover:text-accent transition-colors" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5"
                  d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
              </svg>
              <p class="text-xs text-muted">Drop PDF here or <span class="text-accent font-medium">browse</span></p>
            </div>
            <div id="docInfo" class="hidden">
              <div class="flex items-center justify-center gap-2">
                <svg class="w-5 h-5 text-accent" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                    d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                <span id="docName" class="text-sm font-medium text-gray-900 truncate"></span>
              </div>
              <p id="docPages" class="text-xs text-muted mt-1"></p>
            </div>
            <input type="file" id="fileInput" class="hidden" accept=".pdf" />
          </div>
        </div>

        <!-- Buttons Row -->
        <div class="flex gap-2">
          <button id="btnLoadTestSet" class="flex-1 text-xs font-medium px-3 py-2 rounded-lg border border-border text-gray-300 hover:bg-card2 hover:border-accent/40 transition-all">
            Load Test Set
          </button>
          <button id="btnAddQuery" class="flex-1 text-xs font-medium px-3 py-2 rounded-lg bg-accent/15 text-accent hover:bg-accent/25 transition-all">
            + Add Query
          </button>
        </div>
        <input type="file" id="testSetInput" class="hidden" accept=".json" />

        <!-- Inline Add Form -->
        <div id="addQueryForm" class="hidden bg-card rounded-lg p-4 space-y-3 border border-border animate-fade-in-up">
          <label class="text-[11px] font-medium text-muted uppercase tracking-wider block">New Query</label>
          <input id="newQueryText" type="text" placeholder="Query text..."
            class="w-full text-sm bg-white border border-border rounded-md px-3 py-2 text-gray-900 placeholder-muted focus:outline-none focus:border-accent transition-colors" />
          <div class="flex gap-2">
            <input id="newExpectedPage" type="number" placeholder="Expected pg" min="1"
              class="w-1/2 text-sm bg-white border border-border rounded-md px-3 py-2 text-gray-900 placeholder-muted focus:outline-none focus:border-accent transition-colors" />
            <select id="newQueryType"
              class="w-1/2 text-sm bg-white border border-border rounded-md px-3 py-2 text-gray-900 focus:outline-none focus:border-accent transition-colors">
              <option value="Pinpoint">Pinpoint</option>
              <option value="Reasoning">Reasoning</option>
              <option value="Deep Clause">Deep Clause</option>
              <option value="Negative">Negative</option>
            </select>
          </div>
          <textarea id="newGroundTruth" placeholder="Ground truth answer..."
            class="w-full text-sm bg-white border border-border rounded-md px-3 py-2 text-gray-900 placeholder-muted h-20 resize-none focus:outline-none focus:border-accent transition-colors"></textarea>
          <div class="flex gap-2">
            <button id="btnSaveQuery" class="flex-1 text-xs font-medium px-3 py-2 rounded-lg bg-accent text-white hover:bg-accent2 transition-all">
              Save Query
            </button>
            <button id="btnCancelQuery" class="flex-1 text-xs font-medium px-3 py-2 rounded-lg border border-border text-gray-400 hover:bg-card2 transition-all">
              Cancel
            </button>
          </div>
        </div>

        <!-- Query List -->
        <div>
          <label class="text-[11px] font-medium text-muted uppercase tracking-wider mb-2 block">
            Test Queries <span id="queryCount" class="text-accent"></span>
          </label>
          <div id="queryList" class="space-y-1"></div>
        </div>
      </div>
    </aside>

    <!-- ─── CENTER PANEL (70%) ──────────────────────── -->
    <main class="flex-1 flex flex-col overflow-hidden">

      <!-- Top bar -->
      <div class="px-6 py-4 border-b border-border bg-base/80 backdrop-blur-sm">
        <div class="flex gap-3">
          <div class="flex-1 relative">
            <svg class="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
            <input id="searchInput" type="text" placeholder="Enter a query to test retrieval accuracy…"
              class="w-full bg-card border border-border rounded-lg pl-10 pr-4 py-2.5 text-sm text-gray-900 placeholder-muted focus:outline-none focus:border-accent transition-colors shadow-sm" />
          </div>
          <button id="btnRunQuery"
            class="px-5 py-2.5 rounded-lg bg-accent text-white text-sm font-medium hover:bg-accent2 transition-all flex items-center gap-2 whitespace-nowrap">
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            Run Query
          </button>
        </div>
        <p class="text-[11px] text-muted mt-1.5 flex items-center gap-3">
          <span>Press <kbd class="px-1.5 py-0.5 rounded bg-card border border-border text-[10px] font-mono">Enter</kbd> to run</span>
          <span>•</span>
          <span><kbd class="px-1.5 py-0.5 rounded bg-card border border-border text-[10px] font-mono">Ctrl+B</kbd> batch run</span>
        </p>
      </div>

      <!-- Scrollable Results Area -->
      <div id="resultsArea" class="flex-1 overflow-y-auto px-6 py-5 space-y-5">

        <!-- Empty state -->
        <div id="emptyState" class="flex flex-col items-center justify-center h-full text-center">
          <div class="w-16 h-16 rounded-2xl bg-card border border-border flex items-center justify-center mb-4">
            <svg class="w-8 h-8 text-muted" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5"
                d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
            </svg>
          </div>
          <h2 class="text-base font-medium text-gray-400 mb-1">Select or enter a query</h2>
          <p class="text-xs text-muted max-w-xs">Choose a query from the sidebar or type one above to test retrieval accuracy</p>
        </div>

        <!-- Active Query Panel (hidden initially) -->
        <div id="activeQueryPanel" class="hidden animate-fade-in-up">
          <div class="bg-card rounded-xl border border-border p-5">
            <div class="flex items-start justify-between gap-4">
              <div class="flex-1">
                <p class="text-xs text-muted font-medium uppercase tracking-wider mb-1">Active Query</p>
                <p id="activeQueryText" class="text-sm text-gray-900 font-medium leading-relaxed"></p>
              </div>
              <div id="activeExpectedBadge" class="px-3 py-1 rounded-full bg-accent/10 border border-accent/20 text-accent text-xs font-mono font-medium whitespace-nowrap"></div>
            </div>

            <!-- Ground Truth Collapsible -->
            <div class="mt-3 pt-3 border-t border-border">
              <button id="toggleGroundTruth" class="flex items-center gap-1.5 text-xs text-muted hover:text-gray-600 transition-colors">
                <svg id="gtChevron" class="w-3 h-3 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7" />
                </svg>
                Ground Truth Answer
              </button>
              <div id="groundTruthContent" class="collapsible-content">
                <p id="groundTruthText" class="text-xs text-gray-600 mt-2 leading-relaxed font-mono bg-card2 border border-border rounded-lg p-3"></p>
              </div>
            </div>
          </div>
        </div>

        <!-- Results Container -->
        <div id="resultsContainer" class="hidden space-y-3 animate-fade-in-up">
          <div class="flex items-center justify-between">
            <h2 class="text-sm font-semibold text-gray-900 flex items-center gap-2">
              <svg class="w-4 h-4 text-accent" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                  d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
              </svg>
              Retrieved Chunks
            </h2>
            <span id="resultsSummary" class="text-xs text-muted"></span>
          </div>
          <div id="resultCards" class="space-y-3"></div>
        </div>

        <!-- Batch Progress (hidden) -->
        <div id="batchProgress" class="hidden animate-fade-in-up">
          <div class="bg-card rounded-xl border border-border p-5 shadow-sm">
            <div class="flex items-center justify-between mb-3">
              <p class="text-sm font-medium text-gray-900">Batch Evaluation</p>
              <span id="batchCounter" class="text-xs text-muted font-mono"></span>
            </div>
            <div class="w-full bg-card2 rounded-full h-2 overflow-hidden border border-border">
              <div id="batchBar" class="h-full bg-accent rounded-full progress-bar-striped transition-all duration-300"></div>
            </div>
            <p id="batchStatus" class="text-xs text-muted mt-2"></p>
          </div>
        </div>

        <!-- Batch Results Summary (hidden) -->
        <div id="batchSummary" class="hidden animate-fade-in-up">
          <div class="bg-white rounded-xl border border-border p-5 shadow-sm">
            <h3 class="text-sm font-semibold text-gray-900 mb-4">Batch Results</h3>
            <div class="grid grid-cols-3 gap-3 mb-4">
              <div class="bg-card rounded-lg border border-border p-3 text-center">
                <p id="batchCorrect" class="text-2xl font-bold text-correct font-mono">0</p>
                <p class="text-[11px] text-muted mt-0.5">Correct</p>
              </div>
              <div class="bg-card rounded-lg border border-border p-3 text-center">
                <p id="batchNear" class="text-2xl font-bold text-near font-mono">0</p>
                <p class="text-[11px] text-muted mt-0.5">Near Miss</p>
              </div>
              <div class="bg-card rounded-lg border border-border p-3 text-center">
                <p id="batchWrong" class="text-2xl font-bold text-wrong font-mono">0</p>
                <p class="text-[11px] text-muted mt-0.5">Wrong</p>
              </div>
            </div>
            <div id="batchResultsList" class="space-y-2 max-h-[400px] overflow-y-auto"></div>
          </div>
        </div>
      </div>

      <!-- Bottom Bar -->
      <div class="px-6 py-3 border-t border-border bg-base flex items-center justify-between">
        <p class="text-[11px] text-muted">
          <span id="statusText">Ready</span>
        </p>
        <button id="btnRunAll"
          class="px-4 py-2 rounded-lg border border-accent/40 text-accent text-xs font-medium hover:bg-accent/10 transition-all flex items-center gap-2">
          <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
              d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
          </svg>
          Run All Queries
        </button>
      </div>
    </main>
  </div>

  <!-- Toast -->
  <div id="toast" class="toast bg-card border border-border text-white text-sm"></div>

  <!-- ═══════════════ JAVASCRIPT ═══════════════ -->
  <script>
  (function () {
    "use strict";

    // ── State ──────────────────────────────────────────
    const state = {
      document: { name: "", id: "", pages: 0 },
      queries: [],
      activeQueryIndex: -1,
      results: [],
      batchRunning: false,
    };

    // ── DOM refs ────────────────────────────────────────
    const $ = (s) => document.querySelector(s);
    const $$ = (s) => document.querySelectorAll(s);

    const dropZone      = $("#dropZone");
    const fileInput     = $("#fileInput");
    const dropContent   = $("#dropContent");
    const docInfo       = $("#docInfo");
    const docName       = $("#docName");
    const docPages      = $("#docPages");
    const queryList     = $("#queryList");
    const queryCount    = $("#queryCount");
    const searchInput   = $("#searchInput");
    const btnRunQuery   = $("#btnRunQuery");
    const btnRunAll     = $("#btnRunAll");
    const btnAddQuery   = $("#btnAddQuery");
    const btnLoadTestSet = $("#btnLoadTestSet");
    const testSetInput  = $("#testSetInput");
    const addQueryForm  = $("#addQueryForm");
    const emptyState    = $("#emptyState");
    const activePanel   = $("#activeQueryPanel");
    const resultsContainer = $("#resultsContainer");
    const resultCards   = $("#resultCards");
    const batchProgress = $("#batchProgress");
    const batchSummary  = $("#batchSummary");
    const toast         = $("#toast");

    // ── Init ───────────────────────────────────────────
    function init() {
      renderDocInfo();
      renderQueryList();
      bindEvents();
    }

    // ── Render Doc Info ────────────────────────────────
    function renderDocInfo() {
      if (state.document.name) {
        dropContent.classList.add("hidden");
        docInfo.classList.remove("hidden");
        docName.textContent = state.document.name;
        docPages.textContent = state.document.pages + " pages";
      }
    }

    // ── Render Query List ──────────────────────────────
    function renderQueryList() {
      queryCount.textContent = "(" + state.queries.length + ")";
      queryList.innerHTML = "";
      state.queries.forEach((q, i) => {
        const card = document.createElement("div");
        card.className = "query-card rounded-lg px-3 py-2.5 cursor-pointer" +
          (i === state.activeQueryIndex ? " active" : "");
        card.dataset.index = i;
        card.style.animationDelay = (i * 0.05) + "s";
        card.classList.add("animate-slide-in");

        const typeColors = {
          Pinpoint:     "bg-accent/15 text-accent",
          Reasoning:    "bg-blue-500/15 text-blue-400",
          "Deep Clause":"bg-purple-500/15 text-purple-400",
          Negative:     "bg-wrong/15 text-wrong",
        };
        const colorClass = typeColors[q.type] || "bg-gray-500/15 text-gray-400";

        card.innerHTML =
          '<div class="flex items-start justify-between gap-2">' +
            '<p class="text-xs text-gray-700 leading-relaxed flex-1 line-clamp-2">' + escapeHtml(q.text.substring(0, 80)) + (q.text.length > 80 ? "…" : "") + '</p>' +
            '<span class="font-mono text-[11px] font-medium px-2 py-0.5 rounded-full bg-card2 border border-border text-gray-500 whitespace-nowrap">' +
              (q.expectedPage > 0 ? "Pg " + q.expectedPage : "N/A") +
            '</span>' +
          '</div>' +
          '<div class="mt-1.5">' +
            '<span class="text-[10px] font-medium px-2 py-0.5 rounded-full ' + colorClass + '">' + q.type + '</span>' +
          '</div>';

        card.addEventListener("click", () => selectQuery(i));
        queryList.appendChild(card);
      });
    }

    // ── Select Query ───────────────────────────────────
    function selectQuery(index) {
      state.activeQueryIndex = index;
      const q = state.queries[index];
      searchInput.value = q.text;
      renderQueryList();
      showActivePanel(q);
      // Reset results
      resultsContainer.classList.add("hidden");
      resultCards.innerHTML = "";
      batchProgress.classList.add("hidden");
      batchSummary.classList.add("hidden");
      emptyState.classList.add("hidden");
      activePanel.classList.remove("hidden");
      // Close ground truth
      $("#groundTruthContent").classList.remove("open");
      $("#gtChevron").style.transform = "";
    }

    // ── Show Active Panel ──────────────────────────────
    function showActivePanel(q) {
      $("#activeQueryText").textContent = q.text;
      $("#activeExpectedBadge").textContent = q.expectedPage > 0 ? "Expected: Pg " + q.expectedPage : "Expected: N/A";
      $("#groundTruthText").textContent = q.groundTruth;
    }

    // ── Run Query ──────────────────────────────────────
    async function runQuery() {
      const queryText = searchInput.value.trim();
      if (!queryText) return;

      if (!state.document.id) {
        showToast("Please upload a document to proceed.");
        return;
      }

      let q = state.activeQueryIndex >= 0 ? state.queries[state.activeQueryIndex] : 
              state.queries.find((x) => queryText.toLowerCase().includes(x.text.substring(0, 30).toLowerCase()));

      if (!q) {
        q = { id: "adhoc_" + Date.now(), text: queryText, expectedPage: -1, type: "Ad Hoc", groundTruth: "" };
      }

      showActivePanel(q);
      emptyState.classList.add("hidden");
      activePanel.classList.remove("hidden");

      // Replace ground truth temporarily with thinking placeholder if empty
      if (!q.groundTruth) {
         $("#groundTruthText").textContent = "Waiting for model to generate answer...";
         $("#toggleGroundTruth").click(); // open it
      }

      btnRunQuery.disabled = true;
      btnRunQuery.innerHTML =
        '<svg class="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"></path></svg>' +
        " Searching Tree & Generating Answer…";
      $("#statusText").textContent = "LLM is analyzing reasoning tree...";
      
      try {
        const response = await fetch('/api/query', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query: queryText, doc_id: state.document.id })
        });
        
        const data = await response.json();
        
        if (data.error) {
           showToast("Error: " + data.error);
        } else {
           // Success
           $("#groundTruthText").innerHTML = "<b>Final Generated Answer:</b><br/>" + escapeHtml(data.answer);
           displayResults(data.chunks, q.expectedPage);
           $("#statusText").textContent = "Query complete — " + data.chunks.length + " chunks retrieved";
        }
      } catch (e) {
        showToast("Network error executing query.");
      } finally {
        btnRunQuery.disabled = false;
        btnRunQuery.innerHTML =
          '<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" /><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>' +
          " Run Query";
      }
    }

    // ── Display Results ────────────────────────────────
    function displayResults(results, expectedPage) {
      resultsContainer.classList.remove("hidden");
      resultCards.innerHTML = "";

      let correctCount = 0;
      let nearCount = 0;
      let wrongCount = 0;

      results.forEach((r, i) => {
        // Evaluate accuracy if we have an expected page for this ad-hoc logic
        const diff = expectedPage > 0 && r.page !== "N/A" ? Math.abs(r.page - expectedPage) : -1;
        let matchClass, matchIcon, matchText;

        if (expectedPage <= 0 || r.page === "N/A") {
          matchClass = "text-muted"; matchIcon = "—"; matchText = "Evidential Node";
        } else if (diff === 0) {
          matchClass = "text-correct"; matchIcon = "✅"; matchText = "Correct Page";
          correctCount++;
        } else if (diff <= 2) {
          matchClass = "text-near"; matchIcon = "⚠️"; matchText = "Off by " + diff + " page" + (diff > 1 ? "s" : "");
          nearCount++;
        } else {
          matchClass = "text-wrong"; matchIcon = "❌"; matchText = "Wrong Page";
          wrongCount++;
        }

        const card = document.createElement("div");
        card.className = "result-card bg-card rounded-xl border border-border p-4 shadow-sm animate-fade-in-up";
        card.style.animationDelay = (i * 0.1) + "s";

        card.innerHTML =
          '<div class="flex items-start gap-4">' +
            '<div class="flex flex-col items-center gap-1">' +
              '<span class="w-8 h-8 rounded-lg bg-accent/10 border border-accent/20 text-accent text-xs font-bold flex items-center justify-center font-mono">#' + r.rank + '</span>' +
              '<span class="font-mono text-[11px] font-medium text-gray-500">Pg ' + r.page + '</span>' +
            '</div>' +
            '<div class="flex-1 min-w-0">' +
              '<p class="text-xs text-gray-700 leading-relaxed font-mono bg-white border border-border rounded-lg p-3 mb-2">' + escapeHtml(r.text.substring(0, 250)) + '</p>' +
              '<div class="flex items-center gap-2 ' + matchClass + '">' +
                '<span class="text-sm">' + matchIcon + '</span>' +
                '<span class="text-xs font-medium">' + matchText + '</span>' +
              '</div>' +
            '</div>' +
          '</div>';

        resultCards.appendChild(card);
      });

      const summary = [];
      if (correctCount) summary.push(correctCount + " exact page hits");
      if (nearCount)    summary.push(nearCount + " near misses");
      $("#resultsSummary").textContent = summary.join(" · ") || "";

      setTimeout(() => resultsContainer.scrollIntoView({ behavior: "smooth", block: "start" }), 200);
    }

    // ── Batch Run ──────────────────────────────────────
    async function runBatch() {
      if (state.batchRunning) return;
      state.batchRunning = true;
      batchProgress.classList.remove("hidden");
      batchSummary.classList.add("hidden");
      resultsContainer.classList.add("hidden");
      emptyState.classList.add("hidden");
      activePanel.classList.add("hidden");

      const total = state.queries.length;
      let done = 0;
      let results = { correct: 0, near: 0, wrong: 0, details: [] };

      $("#batchCounter").textContent = "0 / " + total;
      $("#batchBar").style.width = "0%";
      $("#batchStatus").textContent = "Running queries…";
      btnRunAll.disabled = true;

      for (let i = 0; i < total; i++) {
        const q = state.queries[i];
        await new Promise((r) => setTimeout(r, 400 + Math.random() * 400));
        done++;

        const mockRes = MOCK_RESULTS[q.id] || generateRandomResults(q.expectedPage);
        const topPage = mockRes[0].page;
        const diff = q.expectedPage > 0 ? Math.abs(topPage - q.expectedPage) : -1;

        let status;
        if (q.expectedPage <= 0) {
          status = "n/a";
        } else if (diff === 0) {
          status = "correct"; results.correct++;
        } else if (diff <= 2) {
          status = "near"; results.near++;
        } else {
          status = "wrong"; results.wrong++;
        }

        results.details.push({ query: q, topPage, status });

        const pct = Math.round((done / total) * 100);
        $("#batchCounter").textContent = done + " / " + total;
        $("#batchBar").style.width = pct + "%";
        $("#batchStatus").textContent = "Processing: " + q.text.substring(0, 50) + "…";
      }

      state.batchRunning = false;
      btnRunAll.disabled = false;
      batchProgress.classList.add("hidden");
      showBatchSummary(results, total);
      showToast(results.correct + "/" + total + " queries completed — " +
        results.correct + " correct, " + results.near + " near, " + results.wrong + " wrong");
      $("#statusText").textContent = "Batch complete";
    }

    // ── Show Batch Summary ─────────────────────────────
    function showBatchSummary(results, total) {
      batchSummary.classList.remove("hidden");
      $("#batchCorrect").textContent = results.correct;
      $("#batchNear").textContent = results.near;
      $("#batchWrong").textContent = results.wrong;

      const list = $("#batchResultsList");
      list.innerHTML = "";
      results.details.forEach((d) => {
        const statusColors = {
          correct: "bg-correct/15 text-correct",
          near:    "bg-near/15 text-near",
          wrong:   "bg-wrong/15 text-wrong",
          "n/a":   "bg-gray-500/15 text-gray-400",
        };
        const statusLabels = {
          correct: "✅ Correct",
          near:    "⚠️ Near",
          wrong:   "❌ Wrong",
          "n/a":   "— N/A",
        };
        const row = document.createElement("div");
        row.className = "flex items-center justify-between bg-card border border-border rounded-lg px-3 py-2";
        row.innerHTML =
          '<p class="text-xs text-gray-700 flex-1 truncate mr-3">' + escapeHtml(d.query.text.substring(0, 60)) + '…</p>' +
          '<div class="flex items-center gap-2 shrink-0">' +
            '<span class="font-mono text-[11px] text-gray-500">Pg ' + d.topPage + '</span>' +
            '<span class="text-[10px] font-medium px-2 py-0.5 rounded-full ' + statusColors[d.status] + '">' + statusLabels[d.status] + '</span>' +
          '</div>';
        list.appendChild(row);
      });
    }

    // ── Toast ──────────────────────────────────────────
    function showToast(message) {
      toast.textContent = message;
      toast.classList.add("show");
      setTimeout(() => toast.classList.remove("show"), 4000);
    }

    // ── Bind Events ────────────────────────────────────
    function bindEvents() {
      // Drop zone
      dropZone.addEventListener("click", () => fileInput.click());
      dropZone.addEventListener("dragover", (e) => { e.preventDefault(); dropZone.classList.add("dragover"); });
      dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
      dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("dragover");
        const file = e.dataTransfer.files[0];
        if (file && file.type === "application/pdf") handleFile(file);
      });
      fileInput.addEventListener("change", (e) => {
        if (e.target.files[0]) handleFile(e.target.files[0]);
      });

      // Search
      searchInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter") runQuery();
      });
      btnRunQuery.addEventListener("click", runQuery);

      // Batch
      btnRunAll.addEventListener("click", runBatch);
      document.addEventListener("keydown", (e) => {
        if (e.ctrlKey && e.key === "b") { e.preventDefault(); runBatch(); }
      });

      // Ground truth toggle
      $("#toggleGroundTruth").addEventListener("click", () => {
        const content = $("#groundTruthContent");
        const chevron = $("#gtChevron");
        content.classList.toggle("open");
        chevron.style.transform = content.classList.contains("open") ? "rotate(90deg)" : "";
      });

      // Add query
      btnAddQuery.addEventListener("click", () => addQueryForm.classList.toggle("hidden"));
      $("#btnCancelQuery").addEventListener("click", () => {
        addQueryForm.classList.add("hidden");
        clearAddForm();
      });
      $("#btnSaveQuery").addEventListener("click", saveNewQuery);

      // Load test set
      btnLoadTestSet.addEventListener("click", () => testSetInput.click());
      testSetInput.addEventListener("change", (e) => {
        const file = e.target.files[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = (ev) => {
          try {
            const data = JSON.parse(ev.target.result);
            if (Array.isArray(data)) {
              state.queries = data.map((q, i) => ({
                id: q.id || "imported_" + i,
                text: q.text || q.query || "",
                expectedPage: q.expectedPage || q.expected_page || -1,
                type: q.type || "Pinpoint",
                groundTruth: q.groundTruth || q.ground_truth || "",
              }));
              state.activeQueryIndex = -1;
              renderQueryList();
              showToast("Loaded " + state.queries.length + " queries from test set");
            }
          } catch (err) {
            showToast("Error parsing JSON: " + err.message);
          }
        };
        reader.readAsText(file);
      });
    }

    // ── Handle File Upload ─────────────────────────────
    async function handleFile(file) {
      state.document.name = file.name;
      dropContent.classList.add("hidden");
      docInfo.classList.remove("hidden");
      docName.textContent = "Uploading -> " + file.name;
      docPages.textContent = "Transferring...";
      showToast("Starting upload for " + file.name);

      const formData = new FormData();
      formData.append("file", file);

      try {
        const uploadRes = await fetch("/api/upload", { method: "POST", body: formData });
        const uploadData = await uploadRes.json();
        
        if (uploadData.error) {
           docName.textContent = "Upload failed";
           docPages.textContent = uploadData.error;
           showToast("Error: " + uploadData.error);
           return;
        }
        
        state.document.id = uploadData.doc_id;
        docName.textContent = "Processing: " + file.name;
        docPages.textContent = "PageIndex is building reasoning tree...";
        
        // Poll for status
        const pollInterval = setInterval(async () => {
           const statRes = await fetch("/api/status?doc_id=" + state.document.id);
           const statData = await statRes.json();
           
           if (statData.error) {
              clearInterval(pollInterval);
              showToast("Status Error: " + statData.error);
           } else if (statData.status === "ready") {
              clearInterval(pollInterval);
              docName.textContent = file.name;
              docPages.textContent = "Tree built & ready for query!";
              showToast("Document processing complete.");
              $("#statusText").textContent = "Tree built and indexed successfully.";
           } else {
              // still processing
              docPages.textContent = "Processing... (Generating nodes)";
           }
        }, 5000); // 5 sec poll
        
      } catch (e) {
        showToast("Network error uploading file.");
      }
    }

    // ── Save New Query ─────────────────────────────────
    function saveNewQuery() {
      const text = $("#newQueryText").value.trim();
      const page = parseInt($("#newExpectedPage").value) || -1;
      const type = $("#newQueryType").value;
      const gt   = $("#newGroundTruth").value.trim();
      if (!text) return;

      state.queries.push({
        id: "manual_" + Date.now(),
        text, expectedPage: page, type, groundTruth: gt,
      });
      clearAddForm();
      addQueryForm.classList.add("hidden");
      renderQueryList();
      showToast("Query added");
    }

    function clearAddForm() {
      $("#newQueryText").value = "";
      $("#newExpectedPage").value = "";
      $("#newQueryType").value = "Pinpoint";
      $("#newGroundTruth").value = "";
    }

    // ── Helpers ────────────────────────────────────────
    function escapeHtml(str) {
      if (!str) return "";
      const div = document.createElement("div");
      div.textContent = str;
      return div.innerHTML.replace(/\n/g, '<br/>');
    }

    // ── Boot ───────────────────────────────────────────
    init();
  })();
  </script>
</body>
</html>
"""

# ─── Run ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from django.core.management import execute_from_command_line
    execute_from_command_line(sys.argv)
