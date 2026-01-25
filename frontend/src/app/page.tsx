'use client';

import { useState, useEffect } from 'react';
import ParticleBackground from '@/components/ParticleBackground';
import Sidebar from '@/components/Sidebar';
import ChatInterface from '@/components/ChatInterface';

export default function Home() {
  const [selectedAgent, setSelectedAgent] = useState<string | null>(null);
  const [isConnected, setIsConnected] = useState(false);

  // Check API connection on mount
  useEffect(() => {
    const checkConnection = async () => {
      try {
        const response = await fetch('http://localhost:8000/health');
        if (response.ok) {
          setIsConnected(true);
        }
      } catch {
        setIsConnected(false);
      }
    };

    checkConnection();
    const interval = setInterval(checkConnection, 30000); // Check every 30s

    return () => clearInterval(interval);
  }, []);

  return (
    <main className="h-screen w-screen overflow-hidden bg-[#030303] flex">
      {/* Animated background */}
      <ParticleBackground />

      {/* Grid overlay */}
      <div className="fixed inset-0 bg-grid pointer-events-none z-0" />

      {/* Gradient orbs */}
      <div className="fixed top-[-20%] left-[-10%] w-[500px] h-[500px] rounded-full bg-purple-600/20 blur-[120px] pointer-events-none" />
      <div className="fixed bottom-[-20%] right-[-10%] w-[500px] h-[500px] rounded-full bg-blue-600/20 blur-[120px] pointer-events-none" />
      <div className="fixed top-[50%] left-[50%] w-[300px] h-[300px] rounded-full bg-cyan-600/10 blur-[100px] pointer-events-none" />

      {/* Main content */}
      <div className="relative z-10 flex w-full h-full">
        {/* Sidebar */}
        <Sidebar
          selectedAgent={selectedAgent}
          onSelectAgent={setSelectedAgent}
          isConnected={isConnected}
        />

        {/* Chat area */}
        <div className="flex-1 flex flex-col">
          {/* Header */}
          <header className="h-16 border-b border-white/5 flex items-center justify-between px-6 glass">
            <div>
              <h2 className="font-semibold text-white">
                {selectedAgent
                  ? `${selectedAgent.charAt(0).toUpperCase() + selectedAgent.slice(1)} Agent`
                  : 'Smart Router'
                }
              </h2>
              <p className="text-xs text-gray-500">
                {selectedAgent
                  ? 'Direct queries to this specialist'
                  : 'Automatically routes to the best agent'
                }
              </p>
            </div>

            <div className="flex items-center gap-2">
              <div className={`
                px-3 py-1.5 rounded-full text-xs font-medium
                ${isConnected
                  ? 'bg-green-500/10 text-green-400 border border-green-500/20'
                  : 'bg-red-500/10 text-red-400 border border-red-500/20'
                }
              `}>
                {isConnected ? '● Live' : '○ Offline'}
              </div>
            </div>
          </header>

          {/* Chat */}
          <div className="flex-1 overflow-hidden">
            <ChatInterface
              selectedAgent={selectedAgent}
              onAgentUsed={(agent) => {
                // Could update UI to show which agent was used
                console.log('Agent used:', agent);
              }}
            />
          </div>
        </div>
      </div>
    </main>
  );
}
