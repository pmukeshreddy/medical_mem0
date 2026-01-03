'use client';

import { useState, useRef, useEffect } from 'react';
import { Send, Loader2, User, Bot, FileText } from 'lucide-react';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

const SAMPLE_PATIENTS = [
  { id: '59d383a6-12c7-5e1d-9617-d09cab35fc9c', name: 'Taylor Haley' },
  { id: 'patient_1', name: 'Patient 1' },
  { id: 'patient_2', name: 'Patient 2' },
];

const STRATEGIES = [
  { id: 'vanilla', name: 'Vanilla' },
  { id: 'enhanced', name: 'Enhanced' },
];

interface Message {
  role: 'user' | 'assistant';
  content: string;
  memories?: Array<{ content: string; memory?: string }>;
  latency_ms?: number;
}

export default function ChatPage() {
  const [selectedPatient, setSelectedPatient] = useState(SAMPLE_PATIENTS[0].id);
  const [selectedStrategy, setSelectedStrategy] = useState('vanilla');
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() || loading) return;

    const userMessage = input.trim();
    setInput('');
    setMessages((prev) => [...prev, { role: 'user', content: userMessage }]);
    setLoading(true);

    try {
      const response = await fetch(`${API_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          patient_id: selectedPatient,
          message: userMessage,
          history: messages.map((m) => ({ role: m.role, content: m.content })),
          strategy: selectedStrategy,
        }),
      });

      if (!response.ok) throw new Error('Chat failed');

      const data = await response.json();
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: data.response,
          memories: data.memories_used,
          latency_ms: data.latency_ms,
        },
      ]);
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: 'Sorry, I encountered an error. Please try again.',
        },
      ]);
    }

    setLoading(false);
  };

  const handleClear = () => {
    setMessages([]);
  };

  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      {/* Header */}
      <div className="text-center mb-6">
        <h1 className="text-2xl font-bold text-gray-900 mb-2">
          Chat with Patient Memory
        </h1>
        <p className="text-gray-600 text-sm">
          Ask questions about the patient using memory-augmented retrieval
        </p>
      </div>

      {/* Controls */}
      <div className="bg-white rounded-lg shadow-sm border p-4 mb-4">
        <div className="flex flex-wrap gap-4 items-center">
          <div>
            <label className="block text-xs font-medium text-gray-500 mb-1">
              Patient
            </label>
            <select
              value={selectedPatient}
              onChange={(e) => {
                setSelectedPatient(e.target.value);
                handleClear();
              }}
              className="border rounded-lg px-3 py-1.5 text-sm"
            >
              {SAMPLE_PATIENTS.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-xs font-medium text-gray-500 mb-1">
              Retrieval Strategy
            </label>
            <select
              value={selectedStrategy}
              onChange={(e) => setSelectedStrategy(e.target.value)}
              className="border rounded-lg px-3 py-1.5 text-sm"
            >
              {STRATEGIES.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.name}
                </option>
              ))}
            </select>
          </div>

          <div className="ml-auto">
            <button
              onClick={handleClear}
              className="text-sm text-gray-500 hover:text-gray-700"
            >
              Clear chat
            </button>
          </div>
        </div>
      </div>

      {/* Chat Container */}
      <div className="bg-white rounded-lg shadow-sm border flex flex-col h-[500px]">
        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {messages.length === 0 && (
            <div className="text-center text-gray-400 py-12">
              <Bot className="w-12 h-12 mx-auto mb-4 opacity-50" />
              <p>Start a conversation about the patient</p>
              <p className="text-sm mt-2">
                Try: "Does this patient have diabetes?" or "What medications are they on?"
              </p>
            </div>
          )}

          {messages.map((message, idx) => (
            <div
              key={idx}
              className={`flex gap-3 ${
                message.role === 'user' ? 'justify-end' : 'justify-start'
              }`}
            >
              {message.role === 'assistant' && (
                <div className="w-8 h-8 rounded-full bg-blue-100 flex items-center justify-center flex-shrink-0">
                  <Bot className="w-4 h-4 text-blue-600" />
                </div>
              )}

              <div
                className={`max-w-[80%] ${
                  message.role === 'user'
                    ? 'bg-blue-600 text-white rounded-lg px-4 py-2'
                    : 'bg-gray-100 rounded-lg px-4 py-2'
                }`}
              >
                <p className="text-sm">{message.content}</p>

                {/* Show memories used */}
                {message.memories && message.memories.length > 0 && (
                  <div className="mt-3 pt-3 border-t border-gray-200">
                    <p className="text-xs font-medium text-gray-500 mb-2 flex items-center gap-1">
                      <FileText className="w-3 h-3" />
                      Memories used:
                    </p>
                    <div className="space-y-1">
                      {message.memories.slice(0, 3).map((mem, midx) => (
                        <p
                          key={midx}
                          className="text-xs text-gray-600 bg-white rounded px-2 py-1"
                        >
                          {(mem.content || mem.memory || '').slice(0, 100)}...
                        </p>
                      ))}
                    </div>
                  </div>
                )}

                {message.latency_ms && (
                  <p className="text-xs text-gray-400 mt-2">
                    {message.latency_ms.toFixed(0)}ms
                  </p>
                )}
              </div>

              {message.role === 'user' && (
                <div className="w-8 h-8 rounded-full bg-gray-200 flex items-center justify-center flex-shrink-0">
                  <User className="w-4 h-4 text-gray-600" />
                </div>
              )}
            </div>
          ))}

          {loading && (
            <div className="flex gap-3">
              <div className="w-8 h-8 rounded-full bg-blue-100 flex items-center justify-center">
                <Loader2 className="w-4 h-4 text-blue-600 animate-spin" />
              </div>
              <div className="bg-gray-100 rounded-lg px-4 py-2">
                <p className="text-sm text-gray-500">Thinking...</p>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Input */}
        <div className="border-t p-4">
          <div className="flex gap-2">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSend()}
              placeholder="Ask about the patient..."
              className="flex-1 border rounded-lg px-4 py-2 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              disabled={loading}
            />
            <button
              onClick={handleSend}
              disabled={loading || !input.trim()}
              className="bg-blue-600 text-white rounded-lg px-4 py-2 hover:bg-blue-700 disabled:bg-gray-400"
            >
              <Send className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}