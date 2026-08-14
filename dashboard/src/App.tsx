import React, { useEffect, useState } from 'react';
import { Camera, AlertTriangle, Activity, CheckCircle, Settings, Plus, Trash2, Edit3, Send, Eye, EyeOff, RefreshCw, Bell, BarChart2, Video, X } from 'lucide-react';

const API_BASE = "/api";

export default function App() {
  const [activeTab, setActiveTab] = useState('live');

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 font-sans flex">

      {/* Sidebar */}
      <aside className="w-20 lg:w-64 bg-gray-900 border-r border-gray-800 flex flex-col transition-all">
        <div className="p-4 flex items-center justify-center lg:justify-start gap-3 border-b border-gray-800 h-16">
          <Camera className="text-blue-500 w-8 h-8 flex-shrink-0" />
          <h1 className="font-bold text-xl hidden lg:block tracking-wide">CCTV AI</h1>
        </div>

        <nav className="flex-1 py-4 flex flex-col gap-2 px-2">
          <NavItem icon={<Activity />} label="Live View" active={activeTab === 'live'} onClick={() => setActiveTab('live')} />
          <NavItem icon={<AlertTriangle />} label="Events Log" active={activeTab === 'events'} onClick={() => setActiveTab('events')} />
          <NavItem icon={<BarChart2 />} label="Statistics" active={activeTab === 'stats'} onClick={() => setActiveTab('stats')} />
          <NavItem icon={<Camera />} label="Cameras" active={activeTab === 'cameras'} onClick={() => setActiveTab('cameras')} />
          <NavItem icon={<Bell />} label="Telegram" active={activeTab === 'telegram'} onClick={() => setActiveTab('telegram')} />
          <NavItem icon={<Settings />} label="Settings" active={activeTab === 'settings'} onClick={() => setActiveTab('settings')} />
        </nav>
      </aside>

      {/* Main Content */}
      <main className="flex-1 flex flex-col overflow-hidden">
        {/* Header */}
        <header className="h-16 border-b border-gray-800 bg-gray-900/50 backdrop-blur flex items-center px-6 justify-between">
          <h2 className="text-lg font-semibold capitalize">{activeTab === 'live' ? 'Live View' : activeTab === 'cameras' ? 'Camera Management' : activeTab === 'telegram' ? 'Telegram Notifications' : activeTab} Dashboard</h2>
          <div className="flex items-center gap-2">
            <span className="flex h-3 w-3 relative">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-3 w-3 bg-green-500"></span>
            </span>
            <span className="text-sm text-gray-400">System Online</span>
          </div>
        </header>

        {/* Content Area */}
        <div className="flex-1 overflow-auto p-4 lg:p-6">
          {activeTab === 'live' && <LiveView />}
          {activeTab === 'events' && <EventsLog />}
          {activeTab === 'stats' && <StatisticsView />}
          {activeTab === 'cameras' && <CameraManager />}
          {activeTab === 'telegram' && <TelegramConfig />}
          {activeTab === 'settings' && <SettingsView />}
        </div>
      </main>
    </div>
  );
}

function NavItem({ icon, label, active, onClick }: { icon: React.ReactNode, label: string, active: boolean, onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className={`flex items-center gap-3 px-3 lg:px-4 py-3 rounded-lg transition-colors ${
        active
          ? 'bg-blue-600/10 text-blue-500 font-medium'
          : 'text-gray-400 hover:bg-gray-800 hover:text-gray-200'
      }`}
    >
      <div className="flex-shrink-0">{icon}</div>
      <span className="hidden lg:block">{label}</span>
    </button>
  );
}

// ============================================================
// STATISTICS VIEW
// ============================================================
function StatisticsView() {
  const [stats, setStats] = useState<any>(null);

  useEffect(() => {
    fetch(`${API_BASE}/stats/today`).then(r => r.json()).then(d => setStats(d)).catch(() => {});
  }, []);

  if (!stats) return <div className="p-8 text-center text-gray-500">Loading statistics...</div>;

  return (
    <div className="space-y-6">
      <h3 className="text-xl font-bold flex items-center gap-2 mb-6">
        <BarChart2 className="text-blue-500" /> Today's Statistics
      </h3>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard title="Total Events" value={stats.total_events} color="text-blue-400" />
        <StatCard title="Suspicious" value={stats.suspicious_events} color="text-yellow-400" />
        <StatCard title="Critical (Recorded)" value={stats.critical_events} color="text-red-400" />
        <StatCard title="Persons Detected" value={stats.person_detections} color="text-green-400" />
        <StatCard title="Line Crossings (In)" value={stats.masuk || 0} color="text-purple-400" />
        <StatCard title="Line Crossings (Out)" value={stats.keluar || 0} color="text-pink-400" />
      </div>
    </div>
  );
}

function StatCard({ title, value, color }: { title: string, value: string | number, color: string }) {
  return (
    <div className="glass-panel p-6 flex flex-col justify-center items-center text-center">
      <h4 className="text-gray-400 text-sm font-medium mb-2">{title}</h4>
      <span className={`text-4xl font-bold ${color}`}>{value}</span>
    </div>
  );
}

// ============================================================
// LIVE VIEW
// ============================================================
function LiveView() {
  const [cameras, setCameras] = useState<any[]>([]);

  useEffect(() => {
    const fetchStatus = () => {
      fetch(`${API_BASE}/status`).then(r => r.json()).then(d => setCameras(d)).catch(() => {});
    };
    fetchStatus();
    const interval = setInterval(fetchStatus, 2000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        {cameras.length === 0 ? (
          <div className="glass-panel p-12 text-center text-gray-500 col-span-full">
            No cameras connected. Go to <b>Cameras</b> tab to add RTSP streams.
          </div>
        ) : cameras.map(cam => (
          <div key={cam.camera_id} className="glass-panel overflow-hidden flex flex-col">
            <div className="p-3 border-b border-gray-700/50 flex justify-between items-center bg-gray-800/80">
              <div className="flex items-center gap-2">
                <Camera size={16} className="text-gray-400" />
                <span className="font-medium text-gray-200">{cam.name}</span>
              </div>
              <div className="flex gap-3 text-xs font-mono">
                <span className={cam.state !== 'NORMAL' ? 'text-red-400' : 'text-green-400'}>{cam.state}</span>
                <span className="text-blue-400">{cam.fps} FPS</span>
                <span className="text-purple-400">{cam.active_tracks} Tracks</span>
              </div>
            </div>
            <div className="relative bg-black aspect-video flex items-center justify-center">
              <img
                src={`${API_BASE}/stream/${cam.camera_id}`}
                alt={`Stream ${cam.name}`}
                className="w-full h-full object-contain"
                onError={(e) => { (e.target as HTMLImageElement).style.display = 'none'; }}
              />
              {cam.state !== 'NORMAL' && (
                <div className="absolute top-4 right-4 bg-red-500/80 backdrop-blur text-white px-3 py-1 rounded-full text-sm font-bold animate-pulse flex items-center gap-1">
                  <AlertTriangle size={14} /> ALERT
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ============================================================
// EVENTS LOG
// ============================================================
function EventsLog() {
  const [events, setEvents] = useState<any[]>([]);

  const [playingVideo, setPlayingVideo] = useState<string | null>(null);

  useEffect(() => {
    fetch(`${API_BASE}/events?limit=20`).then(r => r.json()).then(d => setEvents(d)).catch(() => {});
  }, []);

  return (
    <div className="glass-panel p-4 lg:p-6 overflow-hidden flex flex-col h-full relative">
      <h3 className="font-semibold text-lg mb-4 flex items-center gap-2">
        <AlertTriangle className="text-yellow-500" /> Recent Suspicious Events
      </h3>
      <div className="overflow-auto flex-1 pr-2 space-y-4">
        {events.length === 0 ? (
          <div className="text-center py-10 text-gray-500">No events logged yet.</div>
        ) : events.map(evt => (
          <div key={evt.event_id} className="bg-gray-800/40 border border-gray-700/50 rounded-lg p-4 flex flex-col md:flex-row gap-4">
            {evt.has_snapshot ? (
              <div className="w-full md:w-48 aspect-video bg-black rounded-md overflow-hidden flex-shrink-0 relative group">
                <img src={`${API_BASE}/snapshots/${evt.snapshot_id}`} className="w-full h-full object-cover" alt="Snapshot" />
                {evt.has_video && (
                  <div 
                    className="absolute inset-0 bg-black/50 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity cursor-pointer"
                    onClick={() => setPlayingVideo(evt.event_id)}
                  >
                    <Video className="w-8 h-8 text-white" />
                  </div>
                )}
              </div>
            ) : (
              <div className="w-full md:w-48 aspect-video bg-gray-900 rounded-md border border-gray-800 flex items-center justify-center text-gray-600 flex-shrink-0">No Photo</div>
            )}
            <div className="flex-1 flex flex-col justify-between">
              <div>
                <div className="flex justify-between items-start mb-2">
                  <div>
                    <h4 className="font-bold text-red-400">{evt.camera_name}</h4>
                    <p className="text-xs text-gray-400 font-mono">{new Date(evt.started_at).toLocaleString()}</p>
                  </div>
                  <span className={`font-bold px-2 py-1 rounded text-sm ${evt.score >= 80 ? 'bg-red-500/20 text-red-400' : 'bg-yellow-500/20 text-yellow-400'}`}>
                    Score: {evt.score}/100
                  </span>
                </div>
                <div className="mt-3">
                  <p className="text-sm text-gray-300 font-medium mb-1">Triggered Rules:</p>
                  <div className="flex flex-wrap gap-2">
                    {evt.triggered_rules.map((rule: string) => (
                      <span key={rule} className="bg-gray-700 text-xs px-2 py-1 rounded-full text-gray-300">{rule}</span>
                    ))}
                  </div>
                </div>
              </div>
              
              {evt.has_video && (
                <div className="mt-4 flex justify-end">
                  <button 
                    onClick={() => setPlayingVideo(evt.event_id)}
                    className="flex items-center gap-2 px-3 py-1.5 bg-blue-600/20 text-blue-400 hover:bg-blue-600/40 rounded text-sm font-medium transition-colors"
                  >
                    <Video size={16} /> Play Recording
                  </button>
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
      
      {/* Video Modal */}
      {playingVideo && (
        <div className="absolute inset-0 bg-gray-950/90 backdrop-blur z-50 flex flex-col items-center justify-center p-4">
          <div className="bg-gray-900 border border-gray-800 p-4 rounded-xl shadow-2xl max-w-4xl w-full">
            <div className="flex justify-between items-center mb-4">
              <h4 className="text-lg font-bold flex items-center gap-2"><Video className="text-blue-500" /> Event Recording</h4>
              <button onClick={() => setPlayingVideo(null)} className="text-gray-400 hover:text-white bg-gray-800 p-1 rounded-full">
                <X size={20} />
              </button>
            </div>
            <div className="aspect-video bg-black rounded-lg overflow-hidden relative">
              <video 
                src={`${API_BASE}/recordings/${playingVideo}`} 
                controls 
                autoPlay 
                className="w-full h-full"
              >
                Your browser does not support the video tag.
              </video>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ============================================================
// CAMERA MANAGER — Add / Edit / Delete RTSP
// ============================================================
function CameraManager() {
  const [cameras, setCameras] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [showForm, setShowForm] = useState(false);
  const [editMode, setEditMode] = useState<string | null>(null);
  const [formData, setFormData] = useState({ camera_id: '', name: '', rtsp_url: '', enabled: true });
  const [message, setMessage] = useState<{type: string, text: string} | null>(null);

  const fetchCameras = () => {
    setLoading(true);
    fetch(`${API_BASE}/cameras`).then(r => r.json()).then(d => { setCameras(d.cameras || []); setLoading(false); }).catch(() => setLoading(false));
  };

  useEffect(fetchCameras, []);

  const handleSubmit = async () => {
    try {
      const url = editMode ? `${API_BASE}/cameras/${editMode}` : `${API_BASE}/cameras`;
      const method = editMode ? 'PUT' : 'POST';
      const res = await fetch(url, { method, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(formData) });
      const data = await res.json();
      if (res.ok) {
        setMessage({ type: 'success', text: data.message });
        setShowForm(false); setEditMode(null);
        setFormData({ camera_id: '', name: '', rtsp_url: '', enabled: true });
        fetchCameras();
      } else {
        setMessage({ type: 'error', text: data.detail || 'Error saving camera' });
      }
    } catch (e) { setMessage({ type: 'error', text: 'Network error' }); }
  };

  const handleDelete = async (id: string) => {
    if (!confirm(`Delete camera ${id}?`)) return;
    try {
      const res = await fetch(`${API_BASE}/cameras/${id}`, { method: 'DELETE' });
      const data = await res.json();
      if (res.ok) { setMessage({ type: 'success', text: data.message }); fetchCameras(); }
      else { setMessage({ type: 'error', text: data.detail }); }
    } catch (e) { setMessage({ type: 'error', text: 'Network error' }); }
  };

  const startEdit = (cam: any) => {
    setFormData({ camera_id: cam.camera_id, name: cam.name, rtsp_url: cam.rtsp_url, enabled: cam.enabled });
    setEditMode(cam.camera_id);
    setShowForm(true);
  };

  return (
    <div className="space-y-6">
      {/* Status message */}
      {message && (
        <div className={`p-4 rounded-lg text-sm font-medium ${message.type === 'success' ? 'bg-green-500/10 border border-green-500/30 text-green-400' : 'bg-red-500/10 border border-red-500/30 text-red-400'}`}>
          {message.text}
        </div>
      )}

      {/* Add camera button */}
      <div className="flex justify-between items-center">
        <h3 className="text-lg font-semibold">RTSP Camera Sources</h3>
        <button onClick={() => { setShowForm(!showForm); setEditMode(null); setFormData({ camera_id: '', name: '', rtsp_url: '', enabled: true }); }}
          className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-500 rounded-lg text-sm font-medium transition-colors">
          <Plus size={16} /> Add Camera
        </button>
      </div>

      {/* Add/Edit form */}
      {showForm && (
        <div className="glass-panel p-6 space-y-4">
          <h4 className="font-semibold text-blue-400">{editMode ? 'Edit Camera' : 'Add New Camera'}</h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm text-gray-400 mb-1">Camera ID</label>
              <input type="text" value={formData.camera_id} onChange={e => setFormData({ ...formData, camera_id: e.target.value })}
                disabled={!!editMode} placeholder="cam_03"
                className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2.5 text-white placeholder-gray-500 focus:border-blue-500 focus:outline-none disabled:opacity-50" />
            </div>
            <div>
              <label className="block text-sm text-gray-400 mb-1">Camera Name</label>
              <input type="text" value={formData.name} onChange={e => setFormData({ ...formData, name: e.target.value })}
                placeholder="Living Room Camera"
                className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2.5 text-white placeholder-gray-500 focus:border-blue-500 focus:outline-none" />
            </div>
          </div>
          <div>
            <label className="block text-sm text-gray-400 mb-1">RTSP URL</label>
            <input type="text" value={formData.rtsp_url} onChange={e => setFormData({ ...formData, rtsp_url: e.target.value })}
              placeholder="rtsp://admin:password@192.168.1.100:554/stream1"
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2.5 text-white placeholder-gray-500 focus:border-blue-500 focus:outline-none font-mono text-sm" />
          </div>
          <div className="flex items-center gap-3">
            <label className="flex items-center gap-2 cursor-pointer">
              <input type="checkbox" checked={formData.enabled} onChange={e => setFormData({ ...formData, enabled: e.target.checked })}
                className="w-4 h-4 rounded bg-gray-700 border-gray-600 text-blue-500 focus:ring-blue-500" />
              <span className="text-sm text-gray-300">Enabled</span>
            </label>
          </div>
          <div className="flex gap-3">
            <button onClick={handleSubmit} className="px-5 py-2.5 bg-blue-600 hover:bg-blue-500 rounded-lg text-sm font-medium transition-colors">
              {editMode ? 'Save Changes' : 'Add Camera'}
            </button>
            <button onClick={() => { setShowForm(false); setEditMode(null); }} className="px-5 py-2.5 bg-gray-700 hover:bg-gray-600 rounded-lg text-sm font-medium transition-colors">
              Cancel
            </button>
          </div>
        </div>
      )}

      {/* Camera list */}
      <div className="space-y-3">
        {loading ? (
          <div className="text-center py-8 text-gray-500">Loading...</div>
        ) : cameras.length === 0 ? (
          <div className="glass-panel p-8 text-center text-gray-500">No cameras configured yet. Click "Add Camera" to add an RTSP stream.</div>
        ) : cameras.map(cam => (
          <div key={cam.camera_id} className="glass-panel p-4 flex flex-col md:flex-row md:items-center justify-between gap-4">
            <div className="flex-1">
              <div className="flex items-center gap-2 mb-1">
                <Camera size={16} className={cam.enabled ? 'text-green-400' : 'text-gray-600'} />
                <span className="font-semibold text-white">{cam.name}</span>
                <span className="text-xs bg-gray-700 px-2 py-0.5 rounded-full text-gray-400 font-mono">{cam.camera_id}</span>
                {!cam.enabled && <span className="text-xs bg-red-500/20 text-red-400 px-2 py-0.5 rounded-full">Disabled</span>}
              </div>
              <p className="text-sm text-gray-400 font-mono truncate">{cam.rtsp_url}</p>
            </div>
            <div className="flex gap-2">
              <button onClick={() => startEdit(cam)} className="p-2 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors" title="Edit">
                <Edit3 size={16} className="text-blue-400" />
              </button>
              <button onClick={() => handleDelete(cam.camera_id)} className="p-2 bg-gray-700 hover:bg-red-600/30 rounded-lg transition-colors" title="Delete">
                <Trash2 size={16} className="text-red-400" />
              </button>
            </div>
          </div>
        ))}
      </div>

      {cameras.length > 0 && (
        <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-lg p-4 text-sm text-yellow-300">
          ⚠️ Setelah menambah/mengubah/menghapus kamera, <strong>restart aplikasi</strong> (<code>python src/main.py</code>) agar perubahan aktif.
        </div>
      )}
    </div>
  );
}

// ============================================================
// TELEGRAM CONFIG + NOTIFICATION FORMAT PREVIEW
// ============================================================
function TelegramConfig() {
  const [tgConfig, setTgConfig] = useState<any>(null);
  const [botToken, setBotToken] = useState('');
  const [chatId, setChatId] = useState('');
  const [showToken, setShowToken] = useState(false);
  const [message, setMessage] = useState<{type: string, text: string} | null>(null);
  const [testing, setTesting] = useState(false);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    fetch(`${API_BASE}/telegram`).then(r => r.json()).then(d => {
      setTgConfig(d);
      setBotToken(d.bot_token_full || '');
      setChatId(d.chat_id || '');
    }).catch(() => {});
  }, []);

  const handleSave = async () => {
    setSaving(true);
    try {
      const res = await fetch(`${API_BASE}/telegram`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ bot_token: botToken, chat_id: chatId }) });
      const data = await res.json();
      if (res.ok) setMessage({ type: 'success', text: data.message });
      else setMessage({ type: 'error', text: data.detail });
    } catch { setMessage({ type: 'error', text: 'Network error' }); }
    setSaving(false);
  };

  const handleTest = async () => {
    setTesting(true);
    try {
      const res = await fetch(`${API_BASE}/telegram/test`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ bot_token: botToken, chat_id: chatId }) });
      const data = await res.json();
      if (res.ok) setMessage({ type: 'success', text: `✅ Test sent! Bot: @${data.bot_username}` });
      else setMessage({ type: 'error', text: data.detail });
    } catch { setMessage({ type: 'error', text: 'Network error' }); }
    setTesting(false);
  };

  const now = new Date().toLocaleString();

  return (
    <div className="space-y-6 max-w-4xl">
      {/* Status message */}
      {message && (
        <div className={`p-4 rounded-lg text-sm font-medium ${message.type === 'success' ? 'bg-green-500/10 border border-green-500/30 text-green-400' : 'bg-red-500/10 border border-red-500/30 text-red-400'}`}>
          {message.text}
        </div>
      )}

      {/* Config form */}
      <div className="glass-panel p-6 space-y-5">
        <h3 className="text-lg font-semibold flex items-center gap-2"><Bell className="text-blue-400" /> Telegram Bot Configuration</h3>

        {tgConfig?.is_active && (
          <div className="flex items-center gap-2 text-green-400 text-sm bg-green-500/10 border border-green-500/30 rounded-lg px-4 py-2.5">
            <CheckCircle size={16} /> Bot is connected and active
            {tgConfig.stats && <span className="text-gray-400 ml-2">| Sent: {tgConfig.stats.sent_count} | Errors: {tgConfig.stats.error_count}</span>}
          </div>
        )}

        <div>
          <label className="block text-sm text-gray-400 mb-1">Bot Token</label>
          <div className="relative">
            <input type={showToken ? 'text' : 'password'} value={botToken} onChange={e => setBotToken(e.target.value)}
              placeholder="123456789:ABCdefGHIjklMNOpqrSTUvwxYZ"
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2.5 text-white placeholder-gray-500 focus:border-blue-500 focus:outline-none font-mono text-sm pr-12" />
            <button onClick={() => setShowToken(!showToken)} className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-500 hover:text-gray-300">
              {showToken ? <EyeOff size={18} /> : <Eye size={18} />}
            </button>
          </div>
        </div>

        <div>
          <label className="block text-sm text-gray-400 mb-1">Chat ID</label>
          <input type="text" value={chatId} onChange={e => setChatId(e.target.value)}
            placeholder="7538465708"
            className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2.5 text-white placeholder-gray-500 focus:border-blue-500 focus:outline-none font-mono text-sm" />
        </div>

        <div className="flex gap-3 flex-wrap">
          <button onClick={handleSave} disabled={saving}
            className="flex items-center gap-2 px-5 py-2.5 bg-blue-600 hover:bg-blue-500 disabled:bg-blue-800 rounded-lg text-sm font-medium transition-colors">
            {saving ? <RefreshCw size={16} className="animate-spin" /> : <CheckCircle size={16} />} Save Configuration
          </button>
          <button onClick={handleTest} disabled={testing || !botToken || !chatId}
            className="flex items-center gap-2 px-5 py-2.5 bg-green-600 hover:bg-green-500 disabled:bg-green-800 disabled:opacity-50 rounded-lg text-sm font-medium transition-colors">
            {testing ? <RefreshCw size={16} className="animate-spin" /> : <Send size={16} />} Send Test Message
          </button>
        </div>
      </div>

      {/* Notification Format Preview */}
      <div className="glass-panel p-6 space-y-5">
        <h3 className="text-lg font-semibold flex items-center gap-2"><AlertTriangle className="text-yellow-400" /> Notification Format Preview</h3>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Alert Message Preview */}
          <div className="space-y-2">
            <h4 className="text-sm font-semibold text-red-400 uppercase tracking-wide">🚨 Initial Alert</h4>
            <div className="bg-gray-800 border border-gray-700 rounded-lg p-4 font-mono text-sm space-y-1.5">
              <p className="text-red-400 font-bold text-base">🚨 SUSPICIOUS DETECTED</p>
              <p className="text-gray-500">━━━━━━━━━━━━━━━━━━</p>
              <p>📹 <b className="text-white">Camera:</b> <span className="text-blue-300">Front Door Camera</span></p>
              <p>🕐 <b className="text-white">Time:</b> <span className="text-gray-300">{now}</span></p>
              <p>⚠️ <b className="text-white">Score:</b> <span className="text-yellow-400">85/100</span></p>
              <p>🔍 <b className="text-white">Triggers:</b> <span className="text-orange-300">restricted_zone_entry, loitering, fast_approach</span></p>
              <p>👤 <b className="text-white">Track IDs:</b> <span className="text-purple-300">[12, 15]</span></p>
              <p>🔖 <b className="text-white">Event:</b> <span className="text-gray-400">a1b2c3d4</span></p>
              <div className="mt-3 bg-gray-900 border border-gray-700 rounded p-3 text-center text-gray-500 text-xs">
                📸 [Snapshot Photo Attached]
              </div>
            </div>
            <p className="text-xs text-gray-500">Dikirim saat skor mencapai threshold (70/100)</p>
          </div>

          {/* Periodic Snapshot Preview */}
          <div className="space-y-2">
            <h4 className="text-sm font-semibold text-blue-400 uppercase tracking-wide">📸 Periodic Snapshot (setiap 2 menit)</h4>
            <div className="bg-gray-800 border border-gray-700 rounded-lg p-4 font-mono text-sm space-y-1.5">
              <p className="text-blue-400 font-bold text-base">📸 Periodic Snapshot #3</p>
              <p className="text-gray-500">━━━━━━━━━━━━━━━━━━</p>
              <p>📹 <b className="text-white">Camera:</b> <span className="text-blue-300">Front Door Camera</span></p>
              <p>🔖 <b className="text-white">Event:</b> <span className="text-gray-400">a1b2c3d4</span></p>
              <p>🕐 <b className="text-white">Time:</b> <span className="text-gray-300">{now}</span></p>
              <p>⚠️ <b className="text-white">Score:</b> <span className="text-yellow-400">78/100</span></p>
              <div className="mt-3 bg-gray-900 border border-gray-700 rounded p-3 text-center text-gray-500 text-xs">
                📸 [Snapshot Photo Attached]
              </div>
            </div>
            <p className="text-xs text-gray-500">Dikirim selama event masih aktif (setiap 120 detik)</p>
          </div>
        </div>

        {/* Event Resolved Preview */}
        <div className="space-y-2">
          <h4 className="text-sm font-semibold text-green-400 uppercase tracking-wide">✅ Event Resolved</h4>
          <div className="bg-gray-800 border border-gray-700 rounded-lg p-4 font-mono text-sm space-y-1.5 max-w-md">
            <p className="text-green-400 font-bold text-base">✅ Event Resolved</p>
            <p className="text-gray-500">━━━━━━━━━━━━━━━━━━</p>
            <p>📹 <b className="text-white">Camera:</b> <span className="text-blue-300">Front Door Camera</span></p>
            <p>🔖 <b className="text-white">Event:</b> <span className="text-gray-400">a1b2c3d4</span></p>
            <p>🕐 <b className="text-white">Resolved at:</b> <span className="text-gray-300">{now}</span></p>
            <p>⏱️ <b className="text-white">Duration:</b> <span className="text-gray-300">12.5 minutes</span></p>
            <p>📸 <b className="text-white">Snapshots:</b> <span className="text-gray-300">6</span></p>
          </div>
        </div>
      </div>

      {/* Technical Details */}
      <div className="glass-panel p-6 space-y-3">
        <h3 className="text-lg font-semibold flex items-center gap-2"><Settings className="text-gray-400" /> Technical Details</h3>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-sm">
          <div className="bg-gray-800/50 rounded-lg p-4 border border-gray-700/50">
            <p className="text-gray-400 mb-1">Retry Policy</p>
            <p className="text-white font-semibold">3× Exponential Backoff</p>
            <p className="text-xs text-gray-500 mt-1">1s → 2s → 4s delay</p>
          </div>
          <div className="bg-gray-800/50 rounded-lg p-4 border border-gray-700/50">
            <p className="text-gray-400 mb-1">Rate Limit</p>
            <p className="text-white font-semibold">Max 1 msg / 3 detik</p>
            <p className="text-xs text-gray-500 mt-1">Per chat, respects Telegram API limits</p>
          </div>
          <div className="bg-gray-800/50 rounded-lg p-4 border border-gray-700/50">
            <p className="text-gray-400 mb-1">Snapshot Interval</p>
            <p className="text-white font-semibold">Setiap 120 detik</p>
            <p className="text-xs text-gray-500 mt-1">Selama event aktif, best-frame selection</p>
          </div>
        </div>
      </div>
    </div>
  );
}

// ============================================================
// SETTINGS
// ============================================================
function SettingsView() {
  return (
    <div className="glass-panel p-6 max-w-2xl">
      <h3 className="font-semibold text-lg mb-4 flex items-center gap-2">
        <Settings className="text-blue-500" /> System Configuration
      </h3>
      <p className="text-gray-400 mb-6">Advanced rule engine and detection parameters can be configured by editing the YAML files directly.</p>
      <div className="space-y-3">
        <div className="bg-gray-800/50 border border-gray-700 p-4 rounded-lg">
          <h4 className="font-medium text-sm text-gray-300 mb-1">Detection Rules</h4>
          <p className="text-xs text-gray-500 font-mono">config/rules.yaml</p>
        </div>
        <div className="bg-gray-800/50 border border-gray-700 p-4 rounded-lg">
          <h4 className="font-medium text-sm text-gray-300 mb-1">System Parameters</h4>
          <p className="text-xs text-gray-500 font-mono">config/config.yaml</p>
        </div>
        <div className="bg-gray-800/50 border border-gray-700 p-4 rounded-lg">
          <h4 className="font-medium text-sm text-gray-300 mb-1">Backend API</h4>
          <div className="flex items-center gap-2 font-mono text-sm text-green-400 mt-1">
            <CheckCircle size={14} /> http://localhost:8000/api
          </div>
        </div>
      </div>
    </div>
  );
}
