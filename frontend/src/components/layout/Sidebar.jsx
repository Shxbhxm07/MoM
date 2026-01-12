import { useState } from "react";
import { NavLink } from "react-router-dom";
import { FiMic, FiUploadCloud, FiChevronLeft, FiChevronRight, FiSettings } from "react-icons/fi";
import { motion, AnimatePresence } from "framer-motion";

export default function Sidebar() {
    const [collapsed, setCollapsed] = useState(false);

    return (
        <motion.aside
            className="sidebar"
            initial={{ width: 280 }}
            animate={{ width: collapsed ? 80 : 280 }}
            transition={{ type: "spring", stiffness: 300, damping: 30 }}
            style={{
                background: 'rgba(255, 255, 255, 0.6)',
                backdropFilter: 'blur(20px)',
                borderRight: '1px solid rgba(255,255,255,0.5)',
                boxShadow: 'none'
            }}
        >
            <div className="sidebar-header" style={{ borderBottom: 'none', paddingBottom: '20px', marginBottom: '20px' }}>
                <div className="logo-mark" style={{ background: 'linear-gradient(135deg, #799351, #a1c181)', borderRadius: '50%', color: '#fff' }}>A</div>
                <AnimatePresence>
                    {!collapsed && (
                        <motion.div
                            className="logo-text"
                            initial={{ opacity: 0, x: -10 }}
                            animate={{ opacity: 1, x: 0 }}
                            exit={{ opacity: 0, x: -10 }}
                        >
                            <div className="logo-title" style={{ fontSize: '18px', letterSpacing: '-0.5px', color: '#2c3e50' }}>AngelBot</div>
                            <div className="logo-subtitle" style={{ textTransform: 'none', letterSpacing: '0', fontSize: '12px', color: '#7f8c8d' }}>Transcription Studio</div>
                        </motion.div>
                    )}
                </AnimatePresence>

                <button
                    className="sidebar-toggle"
                    onClick={() => setCollapsed(!collapsed)}
                    style={{ background: '#fff', border: 'none', borderRadius: '50%', color: '#799351', boxShadow: '0 4px 12px rgba(0,0,0,0.05)' }}
                >
                    {collapsed ? <FiChevronRight /> : <FiChevronLeft />}
                </button>
            </div>

            <nav className="sidebar-nav">
                <NavItem to="/" icon={<FiMic />} label="Live Session" collapsed={collapsed} />
                <NavItem to="/upload" icon={<FiUploadCloud />} label="File Transcription" collapsed={collapsed} />
                <div className="nav-spacer" />
                <NavItem to="/settings" icon={<FiSettings />} label="Settings" collapsed={collapsed} />
            </nav>
        </motion.aside>
    );
}

function NavItem({ to, icon, label, collapsed }) {
    return (
        <NavLink
            to={to}
            className={({ isActive }) => `nav-item ${isActive ? "active" : ""}`}
        >
            <span className="nav-icon">{icon}</span>
            <AnimatePresence>
                {!collapsed && (
                    <motion.span
                        initial={{ opacity: 0, width: 0 }}
                        animate={{ opacity: 1, width: "auto" }}
                        exit={{ opacity: 0, width: 0 }}
                        className="nav-label"
                    >
                        {label}
                    </motion.span>
                )}
            </AnimatePresence>
        </NavLink>
    );
}
