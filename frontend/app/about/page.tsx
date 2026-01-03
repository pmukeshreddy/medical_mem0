'use client';

import { Database, Cpu, GitBranch, Zap } from 'lucide-react';

export default function AboutPage() {
  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      <div className="text-center mb-12">
        <h1 className="text-3xl font-bold text-gray-900 mb-4">
          About MedMem0
        </h1>
        <p className="text-lg text-gray-600">
          A longitudinal patient memory system demonstrating advanced retrieval strategies
        </p>
      </div>

      {/* Architecture */}
      <div className="bg-white rounded-lg shadow-sm border p-6 mb-8">
        <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
          <GitBranch className="w-5 h-5 text-blue-600" />
          Architecture
        </h2>
        <div className="grid md:grid-cols-3 gap-6">
          <div className="text-center p-4 bg-gray-50 rounded-lg">
            <Database className="w-8 h-8 mx-auto mb-2 text-blue-600" />
            <h3 className="font-medium">Mem0 + Pinecone</h3>
            <p className="text-sm text-gray-600 mt-1">
              Vector storage with memory abstraction
            </p>
          </div>
          <div className="text-center p-4 bg-gray-50 rounded-lg">
            <Cpu className="w-8 h-8 mx-auto mb-2 text-green-600" />
            <h3 className="font-medium">FastAPI Backend</h3>
            <p className="text-sm text-gray-600 mt-1">
              RESTful API with retrieval strategies
            </p>
          </div>
          <div className="text-center p-4 bg-gray-50 rounded-lg">
            <Zap className="w-8 h-8 mx-auto mb-2 text-purple-600" />
            <h3 className="font-medium">Next.js Frontend</h3>
            <p className="text-sm text-gray-600 mt-1">
              Interactive comparison interface
            </p>
          </div>
        </div>
      </div>

      {/* Strategies */}
      <div className="bg-white rounded-lg shadow-sm border p-6 mb-8">
        <h2 className="text-xl font-semibold mb-4">Retrieval Strategies</h2>
        <div className="space-y-4">
          <div className="border-l-4 border-blue-500 pl-4">
            <h3 className="font-medium">Vanilla Dense</h3>
            <p className="text-sm text-gray-600">
              Baseline vector similarity search using Mem0's default behavior
            </p>
          </div>
          <div className="border-l-4 border-green-500 pl-4">
            <h3 className="font-medium">Enhanced</h3>
            <p className="text-sm text-gray-600">
              LLM-powered medical term expansion for improved clinical retrieval
            </p>
          </div>
        </div>
      </div>

      {/* How Enhanced Works */}
      <div className="bg-white rounded-lg shadow-sm border p-6 mb-8">
        <h2 className="text-xl font-semibold mb-4">How Enhanced Retrieval Works</h2>
        <div className="p-4 bg-green-50 rounded-lg">
          <h3 className="font-medium text-green-900">LLM Medical Term Expansion</h3>
          <p className="text-sm text-green-700 mt-1">
            Uses GPT-4o-mini to extract clinical synonyms, abbreviations, and related 
            medical concepts from the query. For example, "diabetes" expands to include 
            "DM", "blood glucose", "HbA1c", "hyperglycemia", etc.
          </p>
        </div>
      </div>

      {/* Data */}
      <div className="bg-white rounded-lg shadow-sm border p-6">
        <h2 className="text-xl font-semibold mb-4">Dataset</h2>
        <div className="grid md:grid-cols-3 gap-4 text-center">
          <div className="p-4 bg-gray-50 rounded-lg">
            <p className="text-3xl font-bold text-blue-600">1,500</p>
            <p className="text-sm text-gray-600">Memories</p>
          </div>
          <div className="p-4 bg-gray-50 rounded-lg">
            <p className="text-3xl font-bold text-green-600">115</p>
            <p className="text-sm text-gray-600">Patients</p>
          </div>
          <div className="p-4 bg-gray-50 rounded-lg">
            <p className="text-3xl font-bold text-purple-600">Synthea</p>
            <p className="text-sm text-gray-600">Data Source</p>
          </div>
        </div>
        <p className="text-sm text-gray-500 mt-4">
          Data generated using Synthea synthetic patient generator (HIPAA-safe)
        </p>
      </div>

      {/* Footer */}
      <div className="text-center mt-8 text-gray-500 text-sm">
        <p>Built for Mem0 Applied AI Engineer position</p>
        <p className="mt-1">
          <a
            href="https://github.com/pmukeshreddy/medical_mem0"
            className="text-blue-600 hover:underline"
            target="_blank"
          >
            View on GitHub
          </a>
        </p>
      </div>
    </div>
  );
}