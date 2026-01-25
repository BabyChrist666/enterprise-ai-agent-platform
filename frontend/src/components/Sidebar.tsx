'use client';

import { motion } from 'framer-motion';
import {
  Sparkles,
  TrendingUp,
  Scale,
  HeartPulse,
  Settings,
  Github,
  ExternalLink,
  Zap,
  ChevronRight
} from 'lucide-react';

interface SidebarProps {
  selectedAgent: string | null;
  onSelectAgent: (agent: string | null) => void;
  isConnected: boolean;
}

const agents = [
  {
    id: null,
    name: 'Auto Router',
    description: 'Smart routing',
    icon: Sparkles,
    color: 'purple',
  },
  {
    id: 'finance',
    name: 'Finance',
    description: 'Markets & Risk',
    icon: TrendingUp,
    color: 'green',
  },
  {
    id: 'legal',
    name: 'Legal',
    description: 'Contracts & Compliance',
    icon: Scale,
    color: 'amber',
  },
  {
    id: 'healthcare',
    name: 'Healthcare',
    description: 'Clinical & Medical',
    icon: HeartPulse,
    color: 'rose',
  },
];

const colorMap: Record<string, { bg: string; text: string; ring: string }> = {
  purple: { bg: 'bg-purple-500/10', text: 'text-purple-400', ring: 'ring-purple-500' },
  green: { bg: 'bg-green-500/10', text: 'text-green-400', ring: 'ring-green-500' },
  amber: { bg: 'bg-amber-500/10', text: 'text-amber-400', ring: 'ring-amber-500' },
  rose: { bg: 'bg-rose-500/10', text: 'text-rose-400', ring: 'ring-rose-500' },
};

export default function Sidebar({ selectedAgent, onSelectAgent, isConnected }: SidebarProps) {
  return (
    <div className="w-72 h-full glass-strong flex flex-col">
      {/* Logo */}
      <div className="p-6 border-b border-white/5">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-purple-600 to-blue-600 flex items-center justify-center glow-purple">
            <Zap className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="font-bold text-white">AI Agents</h1>
            <p className="text-xs text-gray-500">Enterprise Platform</p>
          </div>
        </div>
      </div>

      {/* Connection status */}
      <div className="px-4 py-3 border-b border-white/5">
        <div className="flex items-center gap-2">
          <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'} pulse-glow`} />
          <span className="text-xs text-gray-400">
            {isConnected ? 'Connected to Cohere' : 'Disconnected'}
          </span>
        </div>
      </div>

      {/* Agent selector */}
      <div className="flex-1 p-4 overflow-y-auto">
        <p className="text-xs text-gray-500 uppercase tracking-wider mb-3">Select Agent</p>

        <div className="space-y-2">
          {agents.map((agent) => {
            const isSelected = selectedAgent === agent.id;
            const colors = colorMap[agent.color];
            const Icon = agent.icon;

            return (
              <motion.button
                key={agent.id ?? 'auto'}
                whileHover={{ x: 4 }}
                whileTap={{ scale: 0.98 }}
                onClick={() => onSelectAgent(agent.id)}
                className={`
                  w-full p-3 rounded-xl flex items-center gap-3
                  transition-all duration-200
                  ${isSelected
                    ? `${colors.bg} ring-1 ${colors.ring}`
                    : 'hover:bg-white/5'
                  }
                `}
              >
                <div className={`
                  w-9 h-9 rounded-lg flex items-center justify-center
                  ${isSelected ? colors.bg : 'bg-white/5'}
                `}>
                  <Icon className={`w-5 h-5 ${colors.text}`} />
                </div>
                <div className="flex-1 text-left">
                  <p className={`font-medium ${isSelected ? 'text-white' : 'text-gray-300'}`}>
                    {agent.name}
                  </p>
                  <p className="text-xs text-gray-500">{agent.description}</p>
                </div>
                {isSelected && (
                  <ChevronRight className={`w-4 h-4 ${colors.text}`} />
                )}
              </motion.button>
            );
          })}
        </div>
      </div>

      {/* Footer */}
      <div className="p-4 border-t border-white/5 space-y-2">
        <a
          href="https://github.com/BabyChrist666/enterprise-ai-agent-platform"
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-2 p-2 rounded-lg hover:bg-white/5 transition-colors text-gray-400 hover:text-white"
        >
          <Github className="w-4 h-4" />
          <span className="text-sm">View on GitHub</span>
          <ExternalLink className="w-3 h-3 ml-auto" />
        </a>

        <a
          href="http://localhost:8000/docs"
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-2 p-2 rounded-lg hover:bg-white/5 transition-colors text-gray-400 hover:text-white"
        >
          <Settings className="w-4 h-4" />
          <span className="text-sm">API Docs</span>
          <ExternalLink className="w-3 h-3 ml-auto" />
        </a>

        <div className="pt-2 text-center">
          <p className="text-xs text-gray-600">
            Powered by <span className="text-purple-400">Cohere</span>
          </p>
        </div>
      </div>
    </div>
  );
}
