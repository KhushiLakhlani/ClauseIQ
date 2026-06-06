import { useState } from "react"
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from "recharts"
import { Scale, Brain, Layers, FileText, Loader2, ChevronRight } from "lucide-react"
import axios from "axios"

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface ClassifyResult {
  text: string
  predicted_category: string
  confidence: number
  probabilities: Record<string, number>
}

interface ExplainResult {
  text: string
  predicted_category: string
  confidence: number
  top_features: { feature: string; weight: number; positive: boolean }[]
}

interface ClusterTopic {
  topic_id: number
  topic_label: string
  top_terms: string[]
}

interface ClusterResult {
  n_topics: number
  topics: ClusterTopic[]
  assignments: { text: string; topic_id: number; topic_label: string; top_terms: string[] }[]
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const CATEGORY_COLORS: Record<string, string> = {
  "Termination": "#ef4444",
  "Liability": "#f97316",
  "IP Rights": "#8b5cf6",
  "Governance": "#3b82f6",
  "Payment": "#10b981",
  "Duration": "#06b6d4",
  "Other": "#6b7280",
}

const SAMPLE_CLAUSES = [
  "Either party may terminate this agreement upon 30 days written notice.",
  "The Company shall indemnify and hold harmless the Contractor against all claims and damages.",
  "This agreement shall be governed by the laws of the State of New York.",
  "Licensee is granted a non-exclusive, worldwide license to use the Software.",
  "The initial term shall commence on January 1, 2024 and expire on December 31, 2026.",
  "The annual license fee shall not exceed $50,000, payable in quarterly installments.",
  "All intellectual property developed under this agreement shall be owned by the Company.",
]

const API = axios.create({
  baseURL: import.meta.env.PROD
    ? "https://clauseiq-sq4w.onrender.com/api/v1"
    : "/api/v1"
})

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

export default function App() {
  const [activeTab, setActiveTab] = useState<"classify" | "explain" | "cluster">("classify")

  return (
    <div className="min-h-screen bg-slate-50">
      {/* Header */}
      <header className="bg-white border-b border-slate-200 px-6 py-4">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-indigo-600 flex items-center justify-center">
              <Scale className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-slate-900 tracking-tight">ClauseIQ</h1>
              <p className="text-xs text-slate-400">Legal Contract NLP Analytics</p>
            </div>
          </div>
          <div className="flex items-center gap-1 bg-slate-100 rounded-lg p-1">
            <TabButton
              active={activeTab === "classify"}
              onClick={() => setActiveTab("classify")}
              icon={<FileText className="w-4 h-4" />}
              label="Classify"
            />
            <TabButton
              active={activeTab === "explain"}
              onClick={() => setActiveTab("explain")}
              icon={<Brain className="w-4 h-4" />}
              label="Explain"
            />
            <TabButton
              active={activeTab === "cluster"}
              onClick={() => setActiveTab("cluster")}
              icon={<Layers className="w-4 h-4" />}
              label="Cluster"
            />
          </div>
        </div>
      </header>

      {/* Content */}
      <main className="max-w-7xl mx-auto px-6 py-8">
        {activeTab === "classify" && <ClassifyPanel />}
        {activeTab === "explain" && <ExplainPanel />}
        {activeTab === "cluster" && <ClusterPanel />}
      </main>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Tab Button
// ---------------------------------------------------------------------------

function TabButton({ active, onClick, icon, label }: {
  active: boolean
  onClick: () => void
  icon: React.ReactNode
  label: string
}) {
  return (
    <button
      onClick={onClick}
      className={`flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition-all ${
        active
          ? "bg-white text-indigo-600 shadow-sm"
          : "text-slate-500 hover:text-slate-700"
      }`}
    >
      {icon}
      {label}
    </button>
  )
}

// ---------------------------------------------------------------------------
// Classify Panel
// ---------------------------------------------------------------------------

function ClassifyPanel() {
  const [text, setText] = useState("")
  const [results, setResults] = useState<ClassifyResult[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function classifySingle() {
    if (!text.trim()) return
    setLoading(true)
    setError(null)
    try {
      const { data } = await API.post<ClassifyResult>("/classify", { text })
      setResults(prev => [data, ...prev])
      setText("")
    } catch (e: any) {
      setError(e.response?.data?.detail || e.message)
    } finally {
      setLoading(false)
    }
  }

  async function classifySamples() {
    setLoading(true)
    setError(null)
    try {
      const { data } = await API.post<{ results: ClassifyResult[] }>("/classify/batch", {
        clauses: SAMPLE_CLAUSES,
      })
      setResults(data.results)
    } catch (e: any) {
      setError(e.response?.data?.detail || e.message)
    } finally {
      setLoading(false)
    }
  }

  const categoryCounts: Record<string, number> = {}
  results.forEach(r => {
    categoryCounts[r.predicted_category] = (categoryCounts[r.predicted_category] || 0) + 1
  })
  const chartData = Object.entries(categoryCounts)
    .map(([name, count]) => ({ name, count }))
    .sort((a, b) => b.count - a.count)

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-xl border border-slate-200 p-6">
        <h2 className="text-lg font-semibold text-slate-800 mb-4">Classify Contract Clauses</h2>
        <textarea
          rows={3}
          value={text}
          onChange={e => setText(e.target.value)}
          placeholder="Paste a contract clause here..."
          className="w-full rounded-lg border border-slate-300 px-4 py-3 text-sm text-slate-800 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent resize-none"
        />
        <div className="flex gap-3 mt-4">
          <button
            onClick={classifySingle}
            disabled={loading || !text.trim()}
            className="px-5 py-2.5 rounded-lg bg-indigo-600 text-white text-sm font-semibold hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center gap-2"
          >
            {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <ChevronRight className="w-4 h-4" />}
            Classify
          </button>
          <button
            onClick={classifySamples}
            disabled={loading}
            className="px-5 py-2.5 rounded-lg bg-slate-100 text-slate-700 text-sm font-medium hover:bg-slate-200 disabled:opacity-50 transition-all"
          >
            Try Sample Clauses
          </button>
        </div>
        {error && <p className="mt-3 text-sm text-red-500">{error}</p>}
      </div>

      {results.length > 0 && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-white rounded-xl border border-slate-200 p-6">
            <h3 className="text-sm font-semibold text-slate-500 uppercase tracking-wide mb-4">
              Category Distribution
            </h3>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={chartData} layout="vertical" margin={{ left: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis type="number" tick={{ fontSize: 12, fill: "#94a3b8" }} />
                <YAxis type="category" dataKey="name" tick={{ fontSize: 12, fill: "#475569" }} width={100} />
                <Tooltip contentStyle={{ borderRadius: "8px", border: "1px solid #e2e8f0", fontSize: "13px" }} />
                <Bar dataKey="count" radius={[0, 6, 6, 0]}>
                  {chartData.map((entry) => (
                    <Cell key={entry.name} fill={CATEGORY_COLORS[entry.name] || "#6b7280"} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="bg-white rounded-xl border border-slate-200 p-6">
            <h3 className="text-sm font-semibold text-slate-500 uppercase tracking-wide mb-4">
              Classification Results ({results.length})
            </h3>
            <div className="space-y-3 max-h-[300px] overflow-y-auto pr-2">
              {results.map((r, i) => (
                <div key={i} className="flex items-start gap-3 p-3 rounded-lg bg-slate-50">
                  <span
                    className="shrink-0 mt-0.5 px-2.5 py-1 rounded-full text-xs font-semibold text-white"
                    style={{ backgroundColor: CATEGORY_COLORS[r.predicted_category] || "#6b7280" }}
                  >
                    {r.predicted_category}
                  </span>
                  <div className="min-w-0">
                    <p className="text-sm text-slate-700 line-clamp-2">{r.text}</p>
                    <p className="text-xs text-slate-400 mt-1">
                      {(r.confidence * 100).toFixed(1)}% confidence
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Explain Panel
// ---------------------------------------------------------------------------

function ExplainPanel() {
  const [text, setText] = useState(SAMPLE_CLAUSES[0])
  const [result, setResult] = useState<ExplainResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function explain() {
    if (!text.trim()) return
    setLoading(true)
    setError(null)
    try {
      const { data } = await API.post<ExplainResult>("/explain", { text })
      setResult(data)
    } catch (e: any) {
      setError(e.response?.data?.detail || e.message)
    } finally {
      setLoading(false)
    }
  }

  const maxWeight = result
    ? Math.max(...result.top_features.map(f => Math.abs(f.weight)))
    : 1

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-xl border border-slate-200 p-6">
        <h2 className="text-lg font-semibold text-slate-800 mb-4">Explain Classification</h2>
        <p className="text-sm text-slate-500 mb-4">
          See which words influenced the model's decision using LIME (Local Interpretable Model-Agnostic Explanations).
        </p>
        <textarea
          rows={3}
          value={text}
          onChange={e => setText(e.target.value)}
          placeholder="Paste a clause to explain..."
          className="w-full rounded-lg border border-slate-300 px-4 py-3 text-sm text-slate-800 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent resize-none"
        />
        <button
          onClick={explain}
          disabled={loading || !text.trim()}
          className="mt-4 px-5 py-2.5 rounded-lg bg-indigo-600 text-white text-sm font-semibold hover:bg-indigo-700 disabled:opacity-50 transition-all flex items-center gap-2"
        >
          {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Brain className="w-4 h-4" />}
          {loading ? "Analyzing..." : "Explain"}
        </button>
        {error && <p className="mt-3 text-sm text-red-500">{error}</p>}
      </div>

      {result && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-white rounded-xl border border-slate-200 p-6">
            <h3 className="text-sm font-semibold text-slate-500 uppercase tracking-wide mb-4">Prediction</h3>
            <div className="flex items-center gap-4 mb-4">
              <span
                className="px-4 py-2 rounded-full text-sm font-bold text-white"
                style={{ backgroundColor: CATEGORY_COLORS[result.predicted_category] || "#6b7280" }}
              >
                {result.predicted_category}
              </span>
              <span className="text-2xl font-bold text-slate-800">
                {(result.confidence * 100).toFixed(1)}%
              </span>
            </div>
            <div className="p-4 rounded-lg bg-slate-50 text-sm text-slate-700 leading-relaxed">
              {highlightText(result.text, result.top_features)}
            </div>
          </div>

          <div className="bg-white rounded-xl border border-slate-200 p-6">
            <h3 className="text-sm font-semibold text-slate-500 uppercase tracking-wide mb-4">
              Feature Contributions
            </h3>
            <div className="space-y-2">
              {result.top_features.map((f, i) => (
                <div key={i} className="flex items-center gap-3">
                  <span className="w-20 text-right text-xs font-mono text-slate-500 shrink-0">
                    {f.weight > 0 ? "+" : ""}{f.weight.toFixed(4)}
                  </span>
                  <div className="flex-1 flex items-center gap-2">
                    <div
                      className="h-5 rounded-sm"
                      style={{
                        width: `${(Math.abs(f.weight) / maxWeight) * 100}%`,
                        backgroundColor: f.positive ? "#22c55e" : "#ef4444",
                        minWidth: "4px",
                      }}
                    />
                    <span className="text-sm text-slate-700 font-medium">{f.feature}</span>
                  </div>
                </div>
              ))}
            </div>
            <p className="mt-4 text-xs text-slate-400">
              Green bars push toward the predicted category. Red bars push against it.
            </p>
          </div>
        </div>
      )}
    </div>
  )
}

function highlightText(
  text: string,
  features: { feature: string; weight: number; positive: boolean }[]
) {
  const positiveWords = new Set(features.filter(f => f.positive).map(f => f.feature.toLowerCase()))
  const negativeWords = new Set(features.filter(f => !f.positive).map(f => f.feature.toLowerCase()))

  return text.split(/(\s+)/).map((word, i) => {
    const clean = word.toLowerCase().replace(/[^a-z]/g, "")
    if (positiveWords.has(clean)) {
      return (
        <span key={i} className="bg-green-100 text-green-800 px-0.5 rounded font-semibold">
          {word}
        </span>
      )
    }
    if (negativeWords.has(clean)) {
      return (
        <span key={i} className="bg-red-100 text-red-800 px-0.5 rounded font-semibold">
          {word}
        </span>
      )
    }
    return <span key={i}>{word}</span>
  })
}

// ---------------------------------------------------------------------------
// Cluster Panel
// ---------------------------------------------------------------------------

function ClusterPanel() {
  const [texts, setTexts] = useState(SAMPLE_CLAUSES.join("\n\n"))
  const [method, setMethod] = useState<"kmeans" | "nmf">("nmf")
  const [nTopics, setNTopics] = useState(4)
  const [result, setResult] = useState<ClusterResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function cluster() {
    const clauses = texts.split("\n").map(s => s.trim()).filter(s => s.length > 10)
    if (clauses.length < 2) {
      setError("Need at least 2 clauses to cluster")
      return
    }
    setLoading(true)
    setError(null)
    try {
      const { data } = await API.post<ClusterResult>("/cluster", {
        clauses,
        method,
        n_topics: Math.min(nTopics, clauses.length),
      })
      setResult(data)
    } catch (e: any) {
      setError(e.response?.data?.detail || e.message)
    } finally {
      setLoading(false)
    }
  }

  const TOPIC_COLORS = ["#8b5cf6", "#3b82f6", "#10b981", "#f97316", "#ef4444", "#06b6d4", "#ec4899", "#84cc16"]

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-xl border border-slate-200 p-6">
        <h2 className="text-lg font-semibold text-slate-800 mb-4">Topic Clustering</h2>
        <p className="text-sm text-slate-500 mb-4">
          Paste multiple clauses (one per line or separated by blank lines) to discover topic patterns.
        </p>
        <textarea
          rows={6}
          value={texts}
          onChange={e => setTexts(e.target.value)}
          placeholder="Paste clauses here, one per line..."
          className="w-full rounded-lg border border-slate-300 px-4 py-3 text-sm text-slate-800 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent resize-none font-mono"
        />
        <div className="flex items-center gap-4 mt-4">
          <button
            onClick={cluster}
            disabled={loading}
            className="px-5 py-2.5 rounded-lg bg-indigo-600 text-white text-sm font-semibold hover:bg-indigo-700 disabled:opacity-50 transition-all flex items-center gap-2"
          >
            {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Layers className="w-4 h-4" />}
            {loading ? "Clustering..." : "Run Clustering"}
          </button>
          <select
            value={method}
            onChange={e => setMethod(e.target.value as "kmeans" | "nmf")}
            className="px-3 py-2 rounded-lg border border-slate-300 text-sm text-slate-700"
          >
            <option value="nmf">NMF</option>
            <option value="kmeans">KMeans</option>
          </select>
          <div className="flex items-center gap-2">
            <label className="text-sm text-slate-500">Topics:</label>
            <input
              type="number"
              min={2}
              max={10}
              value={nTopics}
              onChange={e => setNTopics(parseInt(e.target.value) || 2)}
              className="w-16 px-2 py-2 rounded-lg border border-slate-300 text-sm text-slate-700"
            />
          </div>
        </div>
        {error && <p className="mt-3 text-sm text-red-500">{error}</p>}
      </div>

      {result && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-white rounded-xl border border-slate-200 p-6">
            <h3 className="text-sm font-semibold text-slate-500 uppercase tracking-wide mb-4">
              Discovered Topics ({result.n_topics})
            </h3>
            <div className="space-y-4">
              {result.topics.map((topic, i) => (
                <div key={topic.topic_id} className="p-3 rounded-lg bg-slate-50">
                  <div className="flex items-center gap-2 mb-2">
                    <div
                      className="w-3 h-3 rounded-full"
                      style={{ backgroundColor: TOPIC_COLORS[i % TOPIC_COLORS.length] }}
                    />
                    <span className="text-sm font-semibold text-slate-700">{topic.topic_label}</span>
                    <span className="text-xs text-slate-400">
                      ({result.assignments.filter(a => a.topic_id === topic.topic_id).length} clauses)
                    </span>
                  </div>
                  <div className="flex flex-wrap gap-1.5">
                    {topic.top_terms.slice(0, 8).map((term, j) => (
                      <span
                        key={j}
                        className="px-2 py-0.5 rounded-full text-xs font-medium text-white"
                        style={{ backgroundColor: TOPIC_COLORS[i % TOPIC_COLORS.length] + "cc" }}
                      >
                        {term}
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="bg-white rounded-xl border border-slate-200 p-6">
            <h3 className="text-sm font-semibold text-slate-500 uppercase tracking-wide mb-4">
              Clause Assignments
            </h3>
            <div className="space-y-2 max-h-[400px] overflow-y-auto pr-2">
              {result.assignments.map((a, i) => (
                <div key={i} className="flex items-start gap-2 p-2 rounded-lg hover:bg-slate-50">
                  <div
                    className="shrink-0 mt-1 w-2.5 h-2.5 rounded-full"
                    style={{ backgroundColor: TOPIC_COLORS[a.topic_id % TOPIC_COLORS.length] }}
                  />
                  <p className="text-sm text-slate-600 line-clamp-2">{a.text}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}