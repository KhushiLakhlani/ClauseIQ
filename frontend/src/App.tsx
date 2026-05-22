import { useState } from 'react'
import './index.css'

const CATEGORIES = [
  'Termination', 'Liability', 'IP Rights', 'Confidentiality',
  'Payment', 'Governance', 'Duration', 'Other',
] as const

type Category = typeof CATEGORIES[number]

interface ClassifyResult {
  predicted_category: Category
  confidence: number
  probabilities: Record<string, number>
}

export default function App() {
  const [text, setText] = useState('')
  const [result, setResult] = useState<ClassifyResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function classify() {
    if (!text.trim()) return
    setLoading(true)
    setError(null)
    try {
      const res = await fetch('/api/v1/classify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      })
      if (!res.ok) throw new Error(`Server error ${res.status}`)
      setResult(await res.json())
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col items-center py-12 px-4">
      <header className="mb-10 text-center">
        <h1 className="text-4xl font-bold text-gray-900 tracking-tight">ClauseIQ</h1>
        <p className="mt-2 text-gray-500 text-sm">Legal Contract NLP Analytics · CUAD Dataset</p>
      </header>

      <main className="w-full max-w-2xl space-y-6">
        <div className="bg-white rounded-2xl shadow-sm border border-gray-200 p-6">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Paste a contract clause
          </label>
          <textarea
            rows={5}
            value={text}
            onChange={e => setText(e.target.value)}
            placeholder="Either party may terminate this agreement upon 30 days written notice…"
            className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm text-gray-800 focus:outline-none focus:ring-2 focus:ring-indigo-500 resize-none"
          />
          <button
            onClick={classify}
            disabled={loading || !text.trim()}
            className="mt-4 w-full rounded-lg bg-indigo-600 px-4 py-2 text-sm font-semibold text-white hover:bg-indigo-700 disabled:opacity-50 transition"
          >
            {loading ? 'Classifying…' : 'Classify Clause'}
          </button>
          {error && <p className="mt-2 text-xs text-red-500">{error}</p>}
        </div>

        {result && (
          <div className="bg-white rounded-2xl shadow-sm border border-gray-200 p-6 space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium text-gray-500 uppercase tracking-wide">
                Predicted Category
              </span>
              <span className="rounded-full bg-indigo-100 px-3 py-1 text-sm font-semibold text-indigo-700">
                {result.predicted_category}
              </span>
            </div>
            <div>
              <span className="text-xs font-medium text-gray-500 uppercase tracking-wide">
                Confidence
              </span>
              <div className="mt-1 h-2 w-full rounded-full bg-gray-100">
                <div
                  className="h-2 rounded-full bg-indigo-500 transition-all"
                  style={{ width: `${(result.confidence * 100).toFixed(1)}%` }}
                />
              </div>
              <p className="mt-1 text-xs text-gray-400 text-right">
                {(result.confidence * 100).toFixed(1)}%
              </p>
            </div>
            <div>
              <span className="text-xs font-medium text-gray-500 uppercase tracking-wide block mb-2">
                All Probabilities
              </span>
              <ul className="space-y-1">
                {CATEGORIES.map(cat => (
                  <li key={cat} className="flex items-center gap-2 text-xs text-gray-600">
                    <span className="w-28 shrink-0">{cat}</span>
                    <div className="flex-1 h-1.5 rounded-full bg-gray-100">
                      <div
                        className="h-1.5 rounded-full bg-indigo-400"
                        style={{ width: `${((result.probabilities[cat] ?? 0) * 100).toFixed(1)}%` }}
                      />
                    </div>
                    <span className="w-10 text-right text-gray-400">
                      {((result.probabilities[cat] ?? 0) * 100).toFixed(1)}%
                    </span>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        )}
      </main>
    </div>
  )
}
