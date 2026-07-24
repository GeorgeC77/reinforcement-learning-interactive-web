import { lazy } from 'react';
import { HashRouter, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import HomePage from './pages/HomePage';

// Chapter pages are lazy-loaded so the initial bundle stays small and each
// chapter is fetched on demand. Shared code (React, KaTeX, charts) is split
// into vendor chunks by Vite/Rollup for caching.

// Chapter 1: Basic Concepts
const Chapter01OverviewPage = lazy(() => import('./pages/chapters/chapter01/OverviewPage'));
const Chapter01MdpPage = lazy(() => import('./pages/chapters/chapter01/MdpPage'));
const Chapter01PolicyPage = lazy(() => import('./pages/chapters/chapter01/PolicyPage'));
const Chapter01RewardPage = lazy(() => import('./pages/chapters/chapter01/RewardPage'));
const Chapter01ReturnsPage = lazy(() => import('./pages/chapters/chapter01/ReturnsPage'));

// Chapter 2: State Values and Bellman Equation
const Chapter02OverviewPage = lazy(() => import('./pages/chapters/chapter02/OverviewPage'));
const Chapter02BellmanPage = lazy(() => import('./pages/chapters/chapter02/BellmanPage'));
const Chapter02StateValuesPage = lazy(() => import('./pages/chapters/chapter02/StateValuesPage'));
const Chapter02ActionValuesPage = lazy(() => import('./pages/chapters/chapter02/ActionValuesPage'));

// Chapter 3: Bellman Optimality Equation
const Chapter03OverviewPage = lazy(() => import('./pages/chapters/chapter03/OverviewPage'));
const Chapter03BoePage = lazy(() => import('./pages/chapters/chapter03/BoePage'));

// Chapter 4: Value Iteration & Policy Iteration
const Chapter04OverviewPage = lazy(() => import('./pages/chapters/chapter04/OverviewPage'));
const Chapter04AlgorithmsPage = lazy(() => import('./pages/chapters/chapter04/AlgorithmsPage'));
const Chapter04ConvergencePage = lazy(() => import('./pages/chapters/chapter04/ConvergencePage'));

// Chapter 5: Monte Carlo Methods
const Chapter05OverviewPage = lazy(() => import('./pages/chapters/chapter05/OverviewPage'));
const Chapter05MonteCarloPage = lazy(() => import('./pages/chapters/chapter05/MonteCarloPage'));
const Chapter05OffPolicyMCPage = lazy(() => import('./pages/chapters/chapter05/OffPolicyMCPage'));

// Chapter 6: Stochastic Approximation
const Chapter06OverviewPage = lazy(() => import('./pages/chapters/chapter06/OverviewPage'));
const Chapter06SaPage = lazy(() => import('./pages/chapters/chapter06/SaPage'));

// Chapter 7: Temporal-Difference Methods
const Chapter07OverviewPage = lazy(() => import('./pages/chapters/chapter07/OverviewPage'));
const Chapter07TdPage = lazy(() => import('./pages/chapters/chapter07/TdPage'));
const Chapter07TdExtensionsPage = lazy(() => import('./pages/chapters/chapter07/TdExtensionsPage'));

// Chapter 8: Value Function Methods
const Chapter08OverviewPage = lazy(() => import('./pages/chapters/chapter08/OverviewPage'));
const Chapter08FaPage = lazy(() => import('./pages/chapters/chapter08/FaPage'));

// Chapter 9: Policy Gradient Methods
const Chapter09OverviewPage = lazy(() => import('./pages/chapters/chapter09/OverviewPage'));
const Chapter09PgPage = lazy(() => import('./pages/chapters/chapter09/PgPage'));

// Chapter 10: Actor-Critic Methods
const Chapter10OverviewPage = lazy(() => import('./pages/chapters/chapter10/OverviewPage'));
const Chapter10AcPage = lazy(() => import('./pages/chapters/chapter10/AcPage'));

// Chapter 11: Exploration & Planning
const Chapter11OverviewPage = lazy(() => import('./pages/chapters/chapter11/OverviewPage'));
const Chapter11ExplorationPage = lazy(() => import('./pages/chapters/chapter11/ExplorationPage'));

// Chapter 12: Advanced Policy Optimization
const Chapter12OverviewPage = lazy(() => import('./pages/chapters/chapter12/OverviewPage'));
const Chapter12PpoPage = lazy(() => import('./pages/chapters/chapter12/PpoPage'));

function App() {
  return (
    <HashRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<HomePage />} />

          {/* Chapter 1 */}
          <Route path="/ch01/overview" element={<Chapter01OverviewPage />} />
          <Route path="/ch01/mdp" element={<Chapter01MdpPage />} />
          <Route path="/ch01/policy" element={<Chapter01PolicyPage />} />
          <Route path="/ch01/reward" element={<Chapter01RewardPage />} />
          <Route path="/ch01/returns" element={<Chapter01ReturnsPage />} />

          {/* Chapter 2 */}
          <Route path="/ch02/overview" element={<Chapter02OverviewPage />} />
          <Route path="/ch02/bellman" element={<Chapter02BellmanPage />} />
          <Route path="/ch02/state-values" element={<Chapter02StateValuesPage />} />
          <Route path="/ch02/action-values" element={<Chapter02ActionValuesPage />} />

          {/* Chapter 3 */}
          <Route path="/ch03/overview" element={<Chapter03OverviewPage />} />
          <Route path="/ch03/boe" element={<Chapter03BoePage />} />

          {/* Chapter 4 */}
          <Route path="/ch04/overview" element={<Chapter04OverviewPage />} />
          <Route path="/ch04/vi-pi" element={<Chapter04AlgorithmsPage />} />
          <Route path="/ch04/convergence" element={<Chapter04ConvergencePage />} />

          {/* Chapter 5 */}
          <Route path="/ch05/overview" element={<Chapter05OverviewPage />} />
          <Route path="/ch05/mc" element={<Chapter05MonteCarloPage />} />
          <Route path="/ch05/off-policy" element={<Chapter05OffPolicyMCPage />} />

          {/* Chapter 6 */}
          <Route path="/ch06/overview" element={<Chapter06OverviewPage />} />
          <Route path="/ch06/sa" element={<Chapter06SaPage />} />

          {/* Chapter 7 */}
          <Route path="/ch07/overview" element={<Chapter07OverviewPage />} />
          <Route path="/ch07/td" element={<Chapter07TdPage />} />
          <Route path="/ch07/td-ext" element={<Chapter07TdExtensionsPage />} />

          {/* Chapter 8 */}
          <Route path="/ch08/overview" element={<Chapter08OverviewPage />} />
          <Route path="/ch08/fa" element={<Chapter08FaPage />} />

          {/* Chapter 9 */}
          <Route path="/ch09/overview" element={<Chapter09OverviewPage />} />
          <Route path="/ch09/pg" element={<Chapter09PgPage />} />

          {/* Chapter 10 */}
          <Route path="/ch10/overview" element={<Chapter10OverviewPage />} />
          <Route path="/ch10/ac" element={<Chapter10AcPage />} />

          {/* Chapter 11 */}
          <Route path="/ch11/overview" element={<Chapter11OverviewPage />} />
          <Route path="/ch11/exploration" element={<Chapter11ExplorationPage />} />

          {/* Chapter 12 */}
          <Route path="/ch12/overview" element={<Chapter12OverviewPage />} />
          <Route path="/ch12/ppo" element={<Chapter12PpoPage />} />
        </Route>
      </Routes>
    </HashRouter>
  );
}

export default App;
