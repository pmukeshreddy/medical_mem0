'use client';

import { useState } from 'react';
import { Search, Plus, Loader2, Clock, Zap } from 'lucide-react';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// Sample patients (loaded from backend in production)
const SAMPLE_PATIENTS = [
  { id: '59d383a6-12c7-5e1d-9617-d09cab35fc9c', name: 'Taylor21 Haley279' },
  { id: 'patient_1', name: 'Patient 1' },
  { id: 'patient_2', name: 'Patient 2' },
];

const STRATEGIES = [
  { id: 'vanilla', name: 'Vanilla', description: 'Basic dense search' },
  { id: 'hybrid', name: 'Hybrid BM25', description: 'BM25 + dense fusion' },
  { id: 'temporal', name: 'Temporal', description: 'Time decay boost' },
  { id: 'entity', name: 'Entity', description: 'Medical entity filter' },
];

interface Memory {
  id: string;
  content: string;
  memory?: string;
  score?: number;
  metadata?: {
    date?: string;
    encounter_type?: string;
  };
}

interface StrategyResult {
  strategy: string;
  memories: Memory[];
  latency_ms: number;
  loading?: boolean;
  error?: string;
}

export default function Home() {
  const [selectedPatient, setSelectedPatient] = useState(SAMPLE_PATIENTS[0].id);
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<StrategyResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [newMemory, setNewMemory] = useState('');
  const [addingMemory, setAddingMemory] = useState(false);
  const [showAddMemory, setShowAddMemory] = useState(false);

  const handleSearch = async () => {
    if (!query.trim()) return;

    setLoading(true);
    setResults(STRATEGIES.map(s => ({ strategy: s.id, memories: [], latency_ms: 0, loading: true })));

    // Run all strategies in parallel
    const promises = STRATEGIES.map(async (strategy) => {
      try {
        const response = await fetch(`${API_URL}/search`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            patient_id: selectedPatient,
            query: query,
            limit: 5,
            strategy: strategy.id,
          }),
        });

        if (!response.ok) throw new Error('Search failed');

        const data = await response.json();
        return {
          strategy: strategy.id,
          memories: data.memories || [],
          latency_ms: data.latency_ms || 0,
        };
      } catch (error) {
        return {
          strategy: strategy.id,
          memories: [],
          latency_ms: 0,
          error: 'Failed to fetch',
        };
      }
    });

    const strategyResults = await Promise.all(promises);
    setResults(strategyResults);
    setLoading(false);
  };

  const handleAddMemory = async () => {
    if (!newMemory.trim()) return;

    setAddingMemory(true);
    try {
      const response = await fetch(`${API_URL}/patients/${selectedPatient}/memories`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content: newMemory }),
      });

      if (response.ok) {
        setNewMemory('');
        setShowAddMemory(false);
        alert('Memory added successfully!');
      }
    } catch (error) {
      alert('Failed to add memory');
    }
    setAddingMemory(false);
  };

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      {/* Header */}
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">
          MedMem0 - Retrieval Strategy Comparison
        </h1>
        <p className="text-gray-600">
          Compare different memory retrieval strategies for longitudinal patient data
        </p>
      </div>

      {/* Search Section */}
      <div className="bg-white rounded-lg shadow-sm border p-6 mb-6">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-4">
          {/* Patient Select */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Patient
            </label>
            <select
              value={selectedPatient}
              onChange={(e) => setSelectedPatient(e.target.value)}
              className="w-full border rounded-lg px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            >
              {SAMPLE_PATIENTS.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name}
                </option>
              ))}
            </select>
          </div>

          {/* Query Input */}
          <div className="md:col-span-2">
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Query
            </label>
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
              placeholder="e.g., diabetes history, medications, blood pressure"
              className="w-full border rounded-lg px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            />
          </div>

          {/* Search Button */}
          <div className="flex items-end">
            <button
              onClick={handleSearch}
              disabled={loading || !query.trim()}
              className="w-full bg-blue-600 text-white rounded-lg px-4 py-2 hover:bg-blue-700 disabled:bg-gray-400 flex items-center justify-center gap-2"
            >
              {loading ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Search className="w-4 h-4" />
              )}
              Compare
            </button>
          </div>
        </div>

        {/* Add Memory Toggle */}
        <div className="border-t pt-4 mt-4">
          <button
            onClick={() => setShowAddMemory(!showAddMemory)}
            className="text-sm text-blue-600 hover:text-blue-700 flex items-center gap-1"
          >
            <Plus className="w-4 h-4" />
            {showAddMemory ? 'Hide' : 'Add new memory'}
          </button>

          {showAddMemory && (
            <div className="mt-3 flex gap-2">
              <input
                type="text"
                value={newMemory}
                onChange={(e) => setNewMemory(e.target.value)}
                placeholder="e.g., Patient started insulin therapy today..."
                className="flex-1 border rounded-lg px-3 py-2 text-sm"
              />
              <button
                onClick={handleAddMemory}
                disabled={addingMemory || !newMemory.trim()}
                className="bg-green-600 text-white rounded-lg px-4 py-2 text-sm hover:bg-green-700 disabled:bg-gray-400"
              >
                {addingMemory ? 'Adding...' : 'Add'}
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Results Grid */}
      {results.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {STRATEGIES.map((strategy) => {
            const result = results.find((r) => r.strategy === strategy.id);
            const isLoading = result?.loading;
            const memories = result?.memories || [];

            return (
              <div
                key={strategy.id}
                className="bg-white rounded-lg shadow-sm border overflow-hidden"
              >
                {/* Strategy Header */}
                <div className="bg-gray-50 px-4 py-3 border-b">
                  <div className="flex items-center justify-between">
                    <h3 className="font-semibold text-gray-900">{strategy.name}</h3>
                    {result && !isLoading && (
                      <span className="flex items-center text-xs text-gray-500">
                        <Clock className="w-3 h-3 mr-1" />
                        {result.latency_ms.toFixed(0)}ms
                      </span>
                    )}
                  </div>
                  <p className="text-xs text-gray-500">{strategy.description}</p>
                </div>

                {/* Results */}
                <div className="p-4 min-h-[300px]">
                  {isLoading ? (
                    <div className="flex items-center justify-center h-full">
                      <Loader2 className="w-6 h-6 animate-spin text-blue-500" />
                    </div>
                  ) : result?.error ? (
                    <div className="text-red-500 text-sm">{result.error}</div>
                  ) : memories.length === 0 ? (
                    <div className="text-gray-400 text-sm text-center">
                      No results
                    </div>
                  ) : (
                    <div className="space-y-3">
                      {memories.map((memory, idx) => (
                        <div
                          key={memory.id || idx}
                          className="p-3 bg-gray-50 rounded-lg text-sm"
                        >
                          <p className="text-gray-800">
                            {memory.content || memory.memory}
                          </p>
                          <div className="flex items-center gap-2 mt-2 text-xs text-gray-500">
                            {memory.score && (
                              <span className="flex items-center">
                                <Zap className="w-3 h-3 mr-1" />
                                {(memory.score * 100).toFixed(1)}%
                              </span>
                            )}
                            {memory.metadata?.date && (
                              <span>
                                {new Date(memory.metadata.date).toLocaleDateString()}
                              </span>
                            )}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Empty State */}
      {results.length === 0 && (
        <div className="text-center py-12 text-gray-500">
          <Search className="w-12 h-12 mx-auto mb-4 opacity-50" />
          <p>Enter a query and click Compare to see results</p>
          <p className="text-sm mt-2">
            Try: "diabetes", "medications", "blood pressure", "recent visits"
          </p>
        </div>
      )}
    </div>
  );
}
