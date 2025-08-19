import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import MainDashboard from './pages/MainDashboard';
import TrajectoryPage from './pages/TrajectoryPage';
import LaunchWindowsPage from './pages/LaunchWindowsPage';

function App() {
  return (
    <Router>
      <div className="App">
        <Routes>
          <Route path="/" element={<MainDashboard />} />
          <Route path="/trajectory" element={<TrajectoryPage />} />
          <Route path="/windows" element={<LaunchWindowsPage />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;