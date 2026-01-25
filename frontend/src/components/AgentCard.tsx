'use client';

import { motion } from 'framer-motion';
import {
  TrendingUp,
  Scale,
  HeartPulse,
  Sparkles,
  ArrowRight
} from 'lucide-react';

interface AgentCardProps {
  type: 'finance' | 'legal' | 'healthcare' | 'orchestrator';
  name: string;
  description: string;
  tools: string[];
  isActive?: boolean;
  onClick?: () => void;
}

const agentConfig = {
  finance: {
    icon: TrendingUp,
    gradient: 'from-green-500 to-emerald-600',
    glow: 'hover:shadow-green-500/20',
    bgGlow: 'bg-green-500/10',
  },
  legal: {
    icon: Scale,
    gradient: 'from-amber-500 to-orange-600',
    glow: 'hover:shadow-amber-500/20',
    bgGlow: 'bg-amber-500/10',
  },
  healthcare: {
    icon: HeartPulse,
    gradient: 'from-rose-500 to-pink-600',
    glow: 'hover:shadow-rose-500/20',
    bgGlow: 'bg-rose-500/10',
  },
  orchestrator: {
    icon: Sparkles,
    gradient: 'from-purple-500 to-blue-600',
    glow: 'hover:shadow-purple-500/20',
    bgGlow: 'bg-purple-500/10',
  },
};

export default function AgentCard({
  type,
  name,
  description,
  tools,
  isActive,
  onClick,
}: AgentCardProps) {
  const config = agentConfig[type];
  const Icon = config.icon;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      whileHover={{ y: -8, scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
      onClick={onClick}
      className={`
        relative p-6 rounded-2xl cursor-pointer
        glass agent-card overflow-hidden
        ${isActive ? 'ring-2 ring-purple-500' : ''}
        ${config.glow} hover:shadow-2xl
        transition-all duration-300
      `}
    >
      {/* Animated gradient background */}
      <div className={`absolute inset-0 ${config.bgGlow} opacity-0 hover:opacity-100 transition-opacity duration-300`} />

      {/* Icon */}
      <div className={`
        relative w-14 h-14 rounded-xl mb-4
        bg-gradient-to-br ${config.gradient}
        flex items-center justify-center
        shadow-lg
      `}>
        <Icon className="w-7 h-7 text-white" />

        {/* Pulse ring when active */}
        {isActive && (
          <motion.div
            className={`absolute inset-0 rounded-xl bg-gradient-to-br ${config.gradient}`}
            initial={{ scale: 1, opacity: 0.5 }}
            animate={{ scale: 1.5, opacity: 0 }}
            transition={{ duration: 1.5, repeat: Infinity }}
          />
        )}
      </div>

      {/* Content */}
      <h3 className="text-xl font-bold text-white mb-2 relative">{name}</h3>
      <p className="text-gray-400 text-sm mb-4 relative">{description}</p>

      {/* Tools preview */}
      <div className="flex flex-wrap gap-2 mb-4 relative">
        {tools.slice(0, 3).map((tool) => (
          <span
            key={tool}
            className="px-2 py-1 text-xs rounded-full bg-white/5 text-gray-300 border border-white/10"
          >
            {tool}
          </span>
        ))}
        {tools.length > 3 && (
          <span className="px-2 py-1 text-xs rounded-full bg-white/5 text-gray-400">
            +{tools.length - 3} more
          </span>
        )}
      </div>

      {/* Action hint */}
      <div className="flex items-center text-sm text-gray-500 relative group-hover:text-white transition-colors">
        <span>Select agent</span>
        <ArrowRight className="w-4 h-4 ml-1" />
      </div>

      {/* Corner decoration */}
      <div className={`
        absolute top-0 right-0 w-24 h-24
        bg-gradient-to-br ${config.gradient}
        opacity-10 blur-2xl
      `} />
    </motion.div>
  );
}
