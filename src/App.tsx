import { HashRouter, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import HomePage from './pages/HomePage';

// Chapter 1: Basic Concepts
import Chapter01OverviewPage from './pages/chapters/chapter01/OverviewPage';
import Chapter01MdpPage from './pages/chapters/chapter01/MdpPage';
import Chapter01PolicyPage from './pages/chapters/chapter01/PolicyPage';
import Chapter01RewardPage from './pages/chapters/chapter01/RewardPage';
import Chapter01ReturnsPage from './pages/chapters/chapter01/ReturnsPage';

// Chapter 2: State Values and Bellman Equation
import Chapter02OverviewPage from './pages/chapters/chapter02/OverviewPage';
import Chapter02BellmanPage from './pages/chapters/chapter02/BellmanPage';
import Chapter02StateValuesPage from './pages/chapters/chapter02/StateValuesPage';
import Chapter02ActionValuesPage from './pages/chapters/chapter02/ActionValuesPage';

// Chapter 3: Bellman Optimality Equation
import Chapter03OverviewPage from './pages/chapters/chapter03/OverviewPage';
import Chapter03BoePage from './pages/chapters/chapter03/BoePage';

// Chapter 4: Value Iteration & Policy Iteration
import Chapter04OverviewPage from './pages/chapters/chapter04/OverviewPage';
import Chapter04AlgorithmsPage from './pages/chapters/chapter04/AlgorithmsPage';

// Chapter 5: Monte Carlo Methods
import Chapter05OverviewPage from './pages/chapters/chapter05/OverviewPage';
import Chapter05MonteCarloPage from './pages/chapters/chapter05/MonteCarloPage';

// Chapter 6: Stochastic Approximation
import Chapter06OverviewPage from './pages/chapters/chapter06/OverviewPage';
import Chapter06SaPage from './pages/chapters/chapter06/SaPage';

// Chapter 7: Temporal-Difference Methods
import Chapter07OverviewPage from './pages/chapters/chapter07/OverviewPage';
import Chapter07TdPage from './pages/chapters/chapter07/TdPage';

// Chapter 8: Value Function Methods
import Chapter08OverviewPage from './pages/chapters/chapter08/OverviewPage';
import Chapter08FaPage from './pages/chapters/chapter08/FaPage';

// Chapter 9: Policy Gradient Methods
import Chapter09OverviewPage from './pages/chapters/chapter09/OverviewPage';
import Chapter09PgPage from './pages/chapters/chapter09/PgPage';

// Chapter 10: Actor-Critic Methods
import Chapter10OverviewPage from './pages/chapters/chapter10/OverviewPage';
import Chapter10AcPage from './pages/chapters/chapter10/AcPage';

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

          {/* Chapter 5 */}
          <Route path="/ch05/overview" element={<Chapter05OverviewPage />} />
          <Route path="/ch05/mc" element={<Chapter05MonteCarloPage />} />

          {/* Chapter 6 */}
          <Route path="/ch06/overview" element={<Chapter06OverviewPage />} />
          <Route path="/ch06/sa" element={<Chapter06SaPage />} />

          {/* Chapter 7 */}
          <Route path="/ch07/overview" element={<Chapter07OverviewPage />} />
          <Route path="/ch07/td" element={<Chapter07TdPage />} />

          {/* Chapter 8 */}
          <Route path="/ch08/overview" element={<Chapter08OverviewPage />} />
          <Route path="/ch08/fa" element={<Chapter08FaPage />} />

          {/* Chapter 9 */}
          <Route path="/ch09/overview" element={<Chapter09OverviewPage />} />
          <Route path="/ch09/pg" element={<Chapter09PgPage />} />

          {/* Chapter 10 */}
          <Route path="/ch10/overview" element={<Chapter10OverviewPage />} />
          <Route path="/ch10/ac" element={<Chapter10AcPage />} />
        </Route>
      </Routes>
    </HashRouter>
  );
}

export default App;
