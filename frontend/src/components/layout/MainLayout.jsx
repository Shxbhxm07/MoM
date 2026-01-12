import { Outlet } from "react-router-dom";
import FocusSidebar from "./FocusSidebar";
import { Toaster } from "react-hot-toast";

export default function MainLayout() {
    return (
        <div className="app-shell focus-layout" style={{
            display: 'flex',
            height: '100vh',
            overflow: 'hidden',
            background: '#fdfbf7' // Paper-like background
        }}>
            <FocusSidebar />

            <main className="main" style={{
                flex: 1,
                display: 'flex',
                flexDirection: 'column',
                position: 'relative',
                background: 'transparent',
                alignItems: 'center' // Center content
            }}>
                <Outlet />
            </main>

            <Toaster
                position="bottom-right"
                toastOptions={{
                    style: {
                        background: '#2c3e50',
                        color: '#fff',
                        fontFamily: 'serif'
                    },
                }}
            />
        </div>
    );
}
