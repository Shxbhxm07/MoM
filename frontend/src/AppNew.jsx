import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import MainLayout from "./components/layout/MainLayout";
import LivePage from "./pages/LivePage";
import FilePage from "./pages/FilePage";
import "./App.css";

function App() {
    return (
        <BrowserRouter>
            <Routes>
                <Route path="/" element={<MainLayout />}>
                    <Route index element={<Navigate to="/live" replace />} />
                    <Route path="live" element={<LivePage />} />
                    <Route path="upload" element={<FilePage />} />
                    {/* Add more routes here */}
                </Route>
            </Routes>
        </BrowserRouter>
    );
}

export default App;
