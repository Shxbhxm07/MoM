import { NavLink } from "react-router-dom";
import { FiEdit3, FiUpload, FiSettings, FiMenu } from "react-icons/fi";

export default function FocusSidebar() {
    return (
        <aside className="focus-sidebar" style={{
            width: '60px',
            background: '#fff',
            borderRight: '1px solid rgba(0,0,0,0.05)',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            padding: '20px 0',
            height: '100vh',
            zIndex: 50
        }}>
            <div style={{ marginBottom: '40px', color: '#2c3e50' }}>
                <FiMenu size={20} />
            </div>

            <nav style={{ display: 'flex', flexDirection: 'column', gap: '24px', flex: 1 }}>
                <FocusLink to="/" icon={<FiEdit3 size={20} />} label="Write" />
                <FocusLink to="/upload" icon={<FiUpload size={20} />} label="Import" />
            </nav>

            <div style={{ marginTop: 'auto', color: '#95a5a6' }}>
                <FiSettings size={20} />
            </div>
        </aside>
    );
}

function FocusLink({ to, icon, label }) {
    return (
        <NavLink
            to={to}
            className={({ isActive }) => isActive ? "focus-link active" : "focus-link"}
            style={({ isActive }) => ({
                width: '40px',
                height: '40px',
                borderRadius: '12px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: isActive ? '#2c3e50' : '#95a5a6',
                background: isActive ? '#f0f2f5' : 'transparent',
                transition: 'all 0.2s'
            })}
            title={label}
        >
            {icon}
        </NavLink>
    );
}
