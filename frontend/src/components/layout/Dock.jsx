import { NavLink } from "react-router-dom";
import { FiMic, FiUploadCloud, FiSettings, FiHome } from "react-icons/fi";
import { motion } from "framer-motion";

export default function Dock() {
    return (
        <div className="dock-container" style={{
            position: 'fixed',
            bottom: '32px',
            left: '50%',
            transform: 'translateX(-50%)',
            zIndex: 1000,
            display: 'flex',
            justifyContent: 'center'
        }}>
            <motion.nav
                className="dock"
                initial={{ y: 100, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                transition={{ type: "spring", stiffness: 260, damping: 20 }}
                style={{
                    display: 'flex',
                    gap: '12px',
                    padding: '12px 16px',
                    background: 'rgba(255, 255, 255, 0.2)',
                    backdropFilter: 'blur(20px)',
                    border: '1px solid rgba(255, 255, 255, 0.3)',
                    borderRadius: '24px',
                    boxShadow: '0 20px 40px rgba(0, 0, 0, 0.1)'
                }}
            >
                <DockItem to="/" icon={<FiMic />} label="Live" />
                <DockItem to="/upload" icon={<FiUploadCloud />} label="Files" />
                <div style={{ width: '1px', background: 'rgba(255,255,255,0.3)', margin: '0 4px' }} />
                <DockItem to="/settings" icon={<FiSettings />} label="Settings" />
            </motion.nav>
        </div>
    );
}

function DockItem({ to, icon, label }) {
    return (
        <NavLink
            to={to}
            className={({ isActive }) => isActive ? "dock-item active" : "dock-item"}
            style={({ isActive }) => ({
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                width: '56px',
                height: '56px',
                borderRadius: '18px',
                textDecoration: 'none',
                color: isActive ? '#fff' : 'rgba(255,255,255,0.8)',
                background: isActive ? 'linear-gradient(135deg, #799351, #a1c181)' : 'rgba(255,255,255,0.1)',
                boxShadow: isActive ? '0 10px 20px rgba(121, 147, 81, 0.4)' : 'none',
                transition: 'all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1)',
                position: 'relative'
            })}
        >
            <motion.div
                whileHover={{ scale: 1.2, y: -5 }}
                whileTap={{ scale: 0.9 }}
                style={{ fontSize: '24px' }}
            >
                {icon}
            </motion.div>
            {/* Tooltip could go here */}
        </NavLink>
    );
}
