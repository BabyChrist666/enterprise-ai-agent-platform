'use client';

import { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Send,
  Loader2,
  Bot,
  User,
  Sparkles,
  TrendingUp,
  Scale,
  HeartPulse,
  Copy,
  Check,
  RefreshCw
} from 'lucide-react';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  agent?: string;
  timestamp: Date;
  isStreaming?: boolean;
}

interface ChatInterfaceProps {
  selectedAgent: string | null;
  onAgentUsed?: (agent: string) => void;
}

const agentIcons: Record<string, typeof Bot> = {
  finance: TrendingUp,
  legal: Scale,
  healthcare: HeartPulse,
  orchestrator: Sparkles,
};

const agentColors: Record<string, string> = {
  finance: 'text-green-400',
  legal: 'text-amber-400',
  healthcare: 'text-rose-400',
  orchestrator: 'text-purple-400',
};

export default function ChatInterface({ selectedAgent, onAgentUsed }: ChatInterfaceProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input.trim(),
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await fetch('http://localhost:8000/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: userMessage.content,
          agent: selectedAgent,
        }),
      });

      const data = await response.json();

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: data.answer || data.detail || 'Sorry, I encountered an error.',
        agent: data.agents_used?.[0] || selectedAgent || 'orchestrator',
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, assistantMessage]);

      if (data.agents_used?.[0] && onAgentUsed) {
        onAgentUsed(data.agents_used[0]);
      }
    } catch (error) {
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: 'Failed to connect to the API. Make sure the backend is running on http://localhost:8000',
        agent: 'orchestrator',
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const copyToClipboard = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const AgentIcon = ({ agent }: { agent?: string }) => {
    const Icon = agent ? agentIcons[agent] || Bot : Bot;
    const colorClass = agent ? agentColors[agent] || 'text-purple-400' : 'text-purple-400';
    return <Icon className={`w-5 h-5 ${colorClass}`} />;
  };

  return (
    <div className="flex flex-col h-full">
      {/* Messages area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 ? (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex flex-col items-center justify-center h-full text-center"
          >
            <div className="w-20 h-20 rounded-2xl bg-gradient-to-br from-purple-500 to-blue-600 flex items-center justify-center mb-6 glow-purple">
              <Sparkles className="w-10 h-10 text-white" />
            </div>
            <h2 className="text-2xl font-bold gradient-text mb-2">
              Enterprise AI Agent Platform
            </h2>
            <p className="text-gray-400 max-w-md">
              Ask me anything about finance, legal matters, or healthcare.
              I&apos;ll route your question to the right specialist agent.
            </p>

            {/* Example prompts */}
            <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-3 max-w-2xl">
              {[
                { text: 'Analyze risk for AAPL, MSFT, GOOGL portfolio', agent: 'finance' },
                { text: 'Review NDA contract for compliance issues', agent: 'legal' },
                { text: 'Check drug interactions for Metformin and Lisinopril', agent: 'healthcare' },
              ].map((example, i) => (
                <motion.button
                  key={i}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: i * 0.1 }}
                  onClick={() => setInput(example.text)}
                  className="p-3 text-left text-sm rounded-xl glass hover:bg-white/5 transition-colors"
                >
                  <span className={agentColors[example.agent]}>{example.agent}</span>
                  <p className="text-gray-300 mt-1">{example.text}</p>
                </motion.button>
              ))}
            </div>
          </motion.div>
        ) : (
          <AnimatePresence>
            {messages.map((message) => (
              <motion.div
                key={message.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className={`flex gap-3 ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                {message.role === 'assistant' && (
                  <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-purple-500/20 to-blue-500/20 flex items-center justify-center flex-shrink-0 border border-white/10">
                    <AgentIcon agent={message.agent} />
                  </div>
                )}

                <div
                  className={`
                    max-w-[80%] rounded-2xl p-4 relative group
                    ${message.role === 'user'
                      ? 'bg-gradient-to-br from-purple-600 to-blue-600 text-white'
                      : 'glass'
                    }
                  `}
                >
                  {message.role === 'assistant' && message.agent && (
                    <div className={`text-xs font-medium mb-2 ${agentColors[message.agent]}`}>
                      {message.agent.charAt(0).toUpperCase() + message.agent.slice(1)} Agent
                    </div>
                  )}

                  <div className="prose prose-invert prose-sm max-w-none">
                    <pre className="whitespace-pre-wrap font-sans text-sm leading-relaxed">
                      {message.content}
                    </pre>
                  </div>

                  {/* Copy button */}
                  {message.role === 'assistant' && (
                    <button
                      onClick={() => copyToClipboard(message.content, message.id)}
                      className="absolute top-2 right-2 p-2 rounded-lg bg-white/5 opacity-0 group-hover:opacity-100 transition-opacity"
                    >
                      {copiedId === message.id ? (
                        <Check className="w-4 h-4 text-green-400" />
                      ) : (
                        <Copy className="w-4 h-4 text-gray-400" />
                      )}
                    </button>
                  )}
                </div>

                {message.role === 'user' && (
                  <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-purple-600 to-blue-600 flex items-center justify-center flex-shrink-0">
                    <User className="w-5 h-5 text-white" />
                  </div>
                )}
              </motion.div>
            ))}
          </AnimatePresence>
        )}

        {isLoading && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex gap-3"
          >
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-purple-500/20 to-blue-500/20 flex items-center justify-center border border-white/10">
              <Loader2 className="w-5 h-5 text-purple-400 animate-spin" />
            </div>
            <div className="glass rounded-2xl p-4">
              <div className="flex items-center gap-2">
                <div className="flex gap-1">
                  <motion.div
                    animate={{ scale: [1, 1.2, 1] }}
                    transition={{ duration: 0.6, repeat: Infinity, delay: 0 }}
                    className="w-2 h-2 rounded-full bg-purple-500"
                  />
                  <motion.div
                    animate={{ scale: [1, 1.2, 1] }}
                    transition={{ duration: 0.6, repeat: Infinity, delay: 0.2 }}
                    className="w-2 h-2 rounded-full bg-blue-500"
                  />
                  <motion.div
                    animate={{ scale: [1, 1.2, 1] }}
                    transition={{ duration: 0.6, repeat: Infinity, delay: 0.4 }}
                    className="w-2 h-2 rounded-full bg-cyan-500"
                  />
                </div>
                <span className="text-gray-400 text-sm">Thinking...</span>
              </div>
            </div>
          </motion.div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input area */}
      <div className="p-4 border-t border-white/5">
        <form onSubmit={handleSubmit} className="relative">
          <div className="glass-strong rounded-2xl overflow-hidden">
            <textarea
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSubmit(e);
                }
              }}
              placeholder={
                selectedAgent
                  ? `Ask the ${selectedAgent} agent...`
                  : 'Ask anything... I\'ll route to the right agent'
              }
              rows={1}
              className="w-full bg-transparent px-4 py-4 pr-14 text-white placeholder-gray-500 resize-none focus:outline-none"
              style={{ minHeight: '56px', maxHeight: '200px' }}
            />

            <button
              type="submit"
              disabled={!input.trim() || isLoading}
              className={`
                absolute right-3 top-1/2 -translate-y-1/2
                w-10 h-10 rounded-xl
                flex items-center justify-center
                transition-all duration-200
                ${input.trim() && !isLoading
                  ? 'bg-gradient-to-r from-purple-600 to-blue-600 hover:shadow-lg hover:shadow-purple-500/25'
                  : 'bg-white/5 cursor-not-allowed'
                }
              `}
            >
              {isLoading ? (
                <Loader2 className="w-5 h-5 text-white animate-spin" />
              ) : (
                <Send className="w-5 h-5 text-white" />
              )}
            </button>
          </div>

          {selectedAgent && (
            <div className="flex items-center gap-2 mt-2 text-xs text-gray-500">
              <AgentIcon agent={selectedAgent} />
              <span>Routing to {selectedAgent} agent</span>
            </div>
          )}
        </form>
      </div>
    </div>
  );
}
