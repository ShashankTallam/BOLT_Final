import React, { useEffect, useRef, useState } from 'react';
import {
  ArrowLeft,
  BookmarkMinus,
  Camera,
  ChevronDown,
  ChevronUp,
  Download,
  CheckCircle2,
  Info as InfoIcon,
  AlertTriangle,
  History,
  Home,
  Mail,
  Maximize,
  Mic,
  Minimize,
  Pause,
  RotateCcw,
  Save,
  Settings,
  Square,
  Trash2,
  User,
  Volume2,
  Hand,
  Play,
  X,
  RotateCw,
  SkipBack,
} from 'lucide-react';

const LipReadingApp = () => {
  // Config: Toasts and History logging
  const TOAST_POSITION = 'bottom-right'; // 'bottom-right' | 'bottom-left' | 'top-right' | 'top-left'
  const TOAST_THEME = 'dark'; // 'dark' | 'light'
  const TOAST_DURATION = 3000; // ms
  const LOG_HISTORY_DELTAS = true; // when true, log only the new chunk instead of full concatenated text
  const HISTORY_DEDUPE = true; // avoid consecutive identical history entries

  // UI/page state
  const [currentPage, setCurrentPage] = useState('home');
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [currentInterface, setCurrentInterface] = useState('lip-reading'); // 'lip-reading' or 'hand-gesture'
  const [isHandGestureDemo, setIsHandGestureDemo] = useState(false);
  const [isVideoModalOpen, setIsVideoModalOpen] = useState(false);
  const [isControlsVisible, setIsControlsVisible] = useState(false);
  const [videoPlaybackSpeed, setVideoPlaybackSpeed] = useState(1);
  const [isVideoPlaying, setIsVideoPlaying] = useState(false);
  const [isScrolled, setIsScrolled] = useState(false);

  // Live/transcription state
  const [isLive, setIsLive] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [transcribedText, setTranscribedText] = useState('');
  const [confidence, setConfidence] = useState(95);

  // Hand gesture state
  const [gestureConfidence, setGestureConfidence] = useState(0);
  const [isHandGestureLive, setIsHandGestureLive] = useState(false);
  const [isHandGestureProcessing, setIsHandGestureProcessing] = useState(false);
  const [gestureText, setGestureText] = useState('');
  const [savedGestures, setSavedGestures] = useState([]);
  // Track all detected gesture texts
  const [gestureHistory, setGestureHistory] = useState([]);
  // Ensure gestureHistory always reflects the full gestureText as shown on the gesture page
  useEffect(() => {
    if (gestureText) {
      setGestureHistory(prev => {
        // Remove all previous entries for this gestureText
        const filtered = prev.filter(item => item.text !== gestureText);
        // Add the latest entry at the top
        return [
          {
            id: Date.now() + Math.random(),
            text: gestureText,
            timestamp: new Date().toLocaleString(),
            confidence: gestureConfidence.toFixed(1),
          },
          ...filtered
        ];
      });
    }
  }, [gestureText, gestureConfidence, gestureHistory]);

  // Audio/TTS state
  const [isPlaying, setIsPlaying] = useState(false);
  const [speechRate, setSpeechRate] = useState(1);
  const [speechPitch, setSpeechPitch] = useState(1);
  const [speechVolume, setSpeechVolume] = useState(1);
  const [isAudioSettingsOpen, setIsAudioSettingsOpen] = useState(false);
  const [isGestureAudioSettingsOpen, setIsGestureAudioSettingsOpen] = useState(false);

  // History/saves and dialogs
  const [savedTexts, setSavedTexts] = useState([]);
  const [allHistory, setAllHistory] = useState([]);
  const [isHistoryModalOpen, setIsHistoryModalOpen] = useState(false);
  const [isConfirmDialogOpen, setIsConfirmDialogOpen] = useState(false);
  const [confirmMessage, setConfirmMessage] = useState('');
  const [confirmAction, setConfirmAction] = useState(null);
  const [historyFilter, setHistoryFilter] = useState('all'); // 'all', 'lip-reading', 'gesture'
  const [isFilterDropdownOpen, setIsFilterDropdownOpen] = useState(false);
  // UI state: expand/collapse long saved texts
  const [expandedSaved, setExpandedSaved] = useState({});
  // Toast notifications
  const [toasts, setToasts] = useState([]);
  const showToast = (message, type = 'success', timeout = TOAST_DURATION) => {
    const id = Date.now() + Math.random();
    setToasts((prev) => [...prev, { id, message, type }]);
    setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id));
    }, timeout);
  };

  // Refs
  const videoRef = useRef(null);
  const gestureVideoRef = useRef(null);
  const demoVideoRef = useRef(null);
  const streamRef = useRef(null);
  const processingIntervalRef = useRef(null);
  const speechRef = useRef(null);
  const transcribedTextRef = useRef('');
  const [useServerVideo, setUseServerVideo] = useState(false);
  const [cameraAvailable, setCameraAvailable] = useState(false);

  // Camera controls
  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user' },
        audio: false,
      });
      // Attach stream to both possible video elements (lip-reading and gesture)
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        try { videoRef.current.play().catch(()=>{}); } catch(e){}
      }
      if (gestureVideoRef.current) {
        gestureVideoRef.current.srcObject = stream;
        try { gestureVideoRef.current.play().catch(()=>{}); } catch(e){}
      }
      streamRef.current = stream;
      // clear any server-stream fallback
      setUseServerVideo(false);
      setCameraAvailable(true);
    } catch (err) {
      console.error('Error accessing camera:', err);
      // Fallback: use server-side MJPEG stream if available
      setUseServerVideo(true);
    }
  };

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    // clear server fallback when stopping
    setUseServerVideo(false);
    setCameraAvailable(false);
    // clear video elements' srcObject so they don't show frozen frames
    try {
      if (videoRef.current) { videoRef.current.pause(); videoRef.current.srcObject = null; }
    } catch (e) {}
    try {
      if (gestureVideoRef.current) { gestureVideoRef.current.pause(); gestureVideoRef.current.srcObject = null; }
    } catch (e) {}
  };

  // Demo texts and simulated live processing
  const sampleTexts = [
    'Hello, this is a demonstration of lip reading technology.',
    'The system analyzes lip movements to convert speech into text.',
    'Advanced AI algorithms process visual information in real-time.',
    'This technology can help bridge communication barriers.',
    'Machine learning enables accurate interpretation of lip movements.',
    'The future of assistive communication is here with BOLT.',
  ];

  // Hand gesture detection data
  const gestureTypes = [
    'Open Palm',
    'Thumbs Up',
    'Peace Sign',
    'Pointing',
    'Fist',
    'OK Sign',
    'Rock On',
    'Number 1',
    'Number 2',
    'Number 3',
  ];

  const gestureActions = {
    'Open Palm': 'Stop/Wait',
    'Thumbs Up': 'Good/Yes/Approve',
    'Peace Sign': 'Victory/Two',
    'Pointing': 'Attention/Direction',
    'Fist': 'Power/Strength',
    'OK Sign': 'Okay/Perfect',
    'Rock On': 'Cool/Awesome',
    'Number 1': 'One/First',
    'Number 2': 'Two/Second',
    'Number 3': 'Three/Third',
  };

  const startLive = () => {
    if (isLive || processingIntervalRef.current) return;
    setIsLive(true);
    setTranscribedText('');
    processingIntervalRef.current = setInterval(() => {
      setIsProcessing(true);
      setTimeout(() => {
        const randomText = sampleTexts[Math.floor(Math.random() * sampleTexts.length)];
        const prev = transcribedTextRef.current;
        const newText = prev ? `${prev} ${randomText}` : randomText;
        setTranscribedText(newText);
        const logText = LOG_HISTORY_DELTAS ? randomText : newText;
        // Add to history only if different from latest snapshot to avoid duplicates
        setAllHistory((h) => {
          if (HISTORY_DEDUPE && h.length > 0 && h[0].text === logText) return h;
          const historyItem = {
            id: Date.now() + Math.random(),
            text: logText,
            timestamp: new Date().toLocaleString(),
            confidence: (90 + Math.random() * 8).toFixed(1),
            isSaved: false,
            type: 'lip-reading', // Flag to identify lip reading history
          };
          return [historyItem, ...h];
        });
        setIsProcessing(false);
        setConfidence(90 + Math.random() * 8);
      }, 1500 + Math.random() * 1000);
    }, 3000 + Math.random() * 2000);
  };

  const stopLive = () => {
    setIsLive(false);
    setIsProcessing(false);
    if (processingIntervalRef.current) {
      clearInterval(processingIntervalRef.current);
      processingIntervalRef.current = null;
    }
  };

  // Hand gesture detection functions
  const startHandGestureLive = async () => {
    // Start backend detection and show server stream so backend can access the camera
    if (isHandGestureLive || processingIntervalRef.current) return;
    try {
      setIsHandGestureProcessing(true);
      // make sure browser releases the camera so backend can use it
      stopCamera();
      const res = await fetch('http://127.0.0.1:5000/start_hand_gestures');
      const data = await res.json();
      console.log('Backend start response:', data);
      // Show server MJPEG stream (backend owns the camera)
      setUseServerVideo(true);
      setIsHandGestureLive(true);
      setGestureText('');

      // Poll backend for latest recognized gesture
      processingIntervalRef.current = setInterval(async () => {
        try {
          const r = await fetch('http://127.0.0.1:5000/latest_result');
          const j = await r.json();
          if (j.type === 'hand-gesture' && j.text) {
            setGestureText(j.text);
          }
        } catch (e) {
          console.error('Polling latest_result failed:', e);
        }
      }, 1200);
    } catch (err) {
      console.error('Failed to start backend hand gestures:', err);
      showToast('Failed to start backend hand gestures', 'error');
    } finally {
      setIsHandGestureProcessing(false);
    }
  };

  const stopHandGestureLive = React.useCallback(async () => {
    setIsHandGestureLive(false);
    setIsHandGestureProcessing(false);
    if (processingIntervalRef.current) {
      clearInterval(processingIntervalRef.current);
      processingIntervalRef.current = null;
    }
    // Ask backend to stop (creates stop.txt which backend watches)
    try {
      await fetch('http://127.0.0.1:5000/stop_hand_gestures');
    } catch (e) {
      console.error('Failed to send stop to backend:', e);
    }
    // clear server fallback and local camera
    setUseServerVideo(false);
    stopCamera();
  }, []);

  const clearGestureText = () => {
    setGestureText('');
  };

  const saveGestureText = () => {
    if (!gestureText.trim()) return;
    if (savedGestures.some((it) => it.text === gestureText)) {
      showToast('Already saved', 'info');
      return;
    }
    const newSaved = {
      id: Date.now(),
      text: gestureText,
      timestamp: new Date().toLocaleString(),
      confidence: gestureConfidence.toFixed(1),
      isSaved: true,
    };
    setSavedGestures((prev) => [newSaved, ...prev]);
    showToast('Saved to Recent Saves', 'success');
  };

  const unsaveGestureText = (id) => {
    setSavedGestures((s) => s.filter((it) => it.id !== id));
    showToast('Removed from saved', 'info');
  };

  const unsaveAllGestures = () => {
    if (savedGestures.length === 0) return;
    setSavedGestures([]);
    showToast('All saved gestures removed', 'info');
  };

  const downloadGestureText = () => {
    if (!gestureText.trim()) return;
    showToast('Download started', 'info');
    const blob = new Blob([gestureText], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `hand-gesture-text-${new Date().toISOString().slice(0, 10)}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const startDemo = () => {
    setIsVideoModalOpen(true);
  };

  const closeDemo = () => {
    setIsHandGestureDemo(false);
    setGestureText('');
    setIsVideoModalOpen(false);
  };

  const closeVideoModal = () => {
    setIsVideoModalOpen(false);
    if (demoVideoRef.current) {
      demoVideoRef.current.pause();
      demoVideoRef.current.currentTime = 0;
      setIsVideoPlaying(false);
    }
  };

  const handlePlaybackSpeedChange = (speed) => {
    setVideoPlaybackSpeed(speed);
    if (demoVideoRef.current) {
      demoVideoRef.current.playbackRate = speed;
    }
  };

  const handleVideoReplay = () => {
    if (demoVideoRef.current) {
      demoVideoRef.current.currentTime = 0;
      demoVideoRef.current.play();
      setIsVideoPlaying(true);
    }
  };

  const handleVideoPlayPause = () => {
    if (demoVideoRef.current) {
      if (isVideoPlaying) {
        demoVideoRef.current.pause();
        setIsVideoPlaying(false);
      } else {
        demoVideoRef.current.play();
        setIsVideoPlaying(true);
      }
    }
  };

  // Text-to-speech controls
  const speakText = () => {
    if (!transcribedText) return;
    speechSynthesis.cancel();
    const utter = new SpeechSynthesisUtterance(transcribedText);
    utter.rate = speechRate;
    utter.pitch = speechPitch;
    utter.volume = speechVolume;
    utter.onstart = () => setIsPlaying(true);
    utter.onend = () => setIsPlaying(false);
    utter.onerror = () => setIsPlaying(false);
    speechRef.current = utter;
    speechSynthesis.speak(utter);
  };

  const speakGestureText = () => {
    if (!gestureText) return;
    speechSynthesis.cancel();
    const utter = new SpeechSynthesisUtterance(gestureText);
    utter.rate = speechRate;
    utter.pitch = speechPitch;
    utter.volume = speechVolume;
    utter.onstart = () => setIsPlaying(true);
    utter.onend = () => setIsPlaying(false);
    utter.onerror = () => setIsPlaying(false);
    speechRef.current = utter;
    speechSynthesis.speak(utter);
  };

  const stopSpeech = () => {
    speechSynthesis.cancel();
    setIsPlaying(false);
  };

  const clearText = () => {
    setTranscribedText('');
    stopSpeech();
  };

  // Save/history helpers
  const saveText = () => {
    if (!transcribedText.trim()) return;
    // Prevent duplicate saves of the same text
    if (savedTexts.some((it) => it.text === transcribedText)) {
      showToast('Already saved', 'info');
      return;
    }
    const match = allHistory.find((it) => it.text === transcribedText);
    const savedId = match ? match.id : Date.now();
    const newSaved = {
      id: savedId,
      text: transcribedText,
      timestamp: new Date().toLocaleString(),
      confidence: confidence.toFixed(1),
      isSaved: true,
    };
    setSavedTexts((prev) => [newSaved, ...prev]);
    setAllHistory((h) => h.map((it) => (it.id === savedId || it.text === transcribedText ? { ...it, isSaved: true } : it)));
    showToast('Saved to Recent Saves', 'success');
  };

  const unsaveAll = () => {
    if (savedTexts.length === 0) return;
    const ids = new Set(savedTexts.map((it) => it.id));
    setSavedTexts([]);
    setAllHistory((h) => h.map((it) => (ids.has(it.id) ? { ...it, isSaved: false } : it)));
    showToast('All saved texts removed', 'info');
  };

  const unsaveText = (id) => {
    setSavedTexts((s) => s.filter((it) => it.id !== id));
    setAllHistory((h) => h.map((it) => (it.id === id ? { ...it, isSaved: false } : it)));
    showToast('Removed from saved', 'info');
  };

  const deleteFromHistory = (id) => {
    setAllHistory((h) => h.filter((it) => it.id !== id));
    setSavedTexts((s) => s.filter((it) => it.id !== id));
    showToast('Deleted from history', 'info');
  };

  const downloadSingleText = (text, timestamp) => {
    showToast('Download started', 'info');
    const blob = new Blob([text], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `lip-reading-${timestamp.replace(/[/:]/g, '-').replace(/\s/g, '_')}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const downloadText = () => {
    if (!transcribedText.trim()) return;
    showToast('Download started', 'info');
    const blob = new Blob([transcribedText], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `lip-reading-text-${new Date().toISOString().slice(0, 10)}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const toggleFullscreen = () => setIsFullscreen((v) => !v);

  // Confirmation dialog
  const showConfirmDialog = (message, action) => {
    setConfirmMessage(message);
    setConfirmAction(() => action);
    setIsConfirmDialogOpen(true);
  };
  const handleConfirm = () => {
    if (confirmAction) confirmAction();
    setIsConfirmDialogOpen(false);
    setConfirmAction(null);
  };
  const handleCancel = () => {
    setIsConfirmDialogOpen(false);
    setConfirmAction(null);
  };

  // Audio settings
  const toggleAudioSettings = () => setIsAudioSettingsOpen((v) => !v);
  const toggleGestureAudioSettings = () => setIsGestureAudioSettingsOpen((v) => !v);
  const restartIfPlaying = (opts) => {
    if (!(isPlaying && transcribedText)) return;
    stopSpeech();
    setTimeout(() => {
      const utter = new SpeechSynthesisUtterance(transcribedText);
      utter.rate = opts.rate ?? speechRate;
      utter.pitch = opts.pitch ?? speechPitch;
      utter.volume = opts.volume ?? speechVolume;
      utter.onstart = () => setIsPlaying(true);
      utter.onend = () => setIsPlaying(false);
      speechSynthesis.speak(utter);
    }, 100);
  };
  const handleRateChange = (value) => {
    const v = parseFloat(value);
    setSpeechRate(v);
    restartIfPlaying({ rate: v });
  };
  const handlePitchChange = (value) => {
    const v = parseFloat(value);
    setSpeechPitch(v);
    restartIfPlaying({ pitch: v });
  };
  const handleVolumeChange = (value) => {
    const v = parseFloat(value);
    setSpeechVolume(v);
    restartIfPlaying({ volume: v });
  };

  useEffect(() => {
    if (currentPage === 'home') {
      startCamera();
    } else {
      stopCamera();
      stopLive();
      stopHandGestureLive();
      stopSpeech();
      setGestureText('');
    }
    return () => {
      stopCamera();
      stopLive();
      stopHandGestureLive();
      stopSpeech();
    };
  }, [currentPage, stopHandGestureLive]);

  useEffect(() => () => {
    stopLive();
    stopHandGestureLive();
    stopSpeech();
  }, [stopHandGestureLive]);

  // keep a ref of the latest transcribed text for interval handler
  useEffect(() => {
    transcribedTextRef.current = transcribedText;
  }, [transcribedText]);

  // Scroll detection for navbar transparency
  useEffect(() => {
    const handleScroll = () => {
      const scrollTop = window.scrollY;
      setIsScrolled(scrollTop > 50);
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  // Helper function for logo click - navigate to home or scroll to top if already on home
  const handleLogoClick = () => {
    if (currentPage === 'home') {
      window.scrollTo({ top: 0, behavior: 'smooth' });
    } else {
      navigateToPage('home');
    }
  };

  const navigateToPage = (page) => setCurrentPage(page);

  const renderHome = () => (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-black to-gray-800 relative overflow-hidden">
      {/* Enhanced Professional background graphics */}
      <div className="absolute inset-0">
        {/* Large Gradient orbs with enhanced effects */}
        <div className="absolute top-20 left-20 w-96 h-96 bg-gradient-to-r from-orange-500/20 to-amber-500/20 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute bottom-20 right-20 w-80 h-80 bg-gradient-to-r from-orange-400/15 to-yellow-500/15 rounded-full blur-3xl animate-pulse delay-1000"></div>
        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-72 h-72 bg-gradient-to-r from-amber-500/10 to-orange-600/10 rounded-full blur-3xl animate-pulse delay-2000"></div>
        
        {/* Additional floating orbs for depth */}
        <div className="absolute top-40 right-1/4 w-32 h-32 bg-gradient-to-r from-amber-400/25 to-orange-400/25 rounded-full blur-2xl animate-float"></div>
        <div className="absolute bottom-40 left-1/3 w-28 h-28 bg-gradient-to-r from-orange-300/20 to-yellow-400/20 rounded-full blur-2xl animate-float-delay"></div>
        <div className="absolute top-3/4 left-1/4 w-24 h-24 bg-gradient-to-r from-amber-600/15 to-orange-500/15 rounded-full blur-xl animate-float-slow"></div>
        
        {/* Advanced Geometric patterns */}
        <div className="absolute top-32 right-32 w-64 h-64 border border-orange-500/20 rounded-lg transform rotate-45 animate-slow-spin"></div>
        <div className="absolute bottom-32 left-32 w-48 h-48 border-2 border-dashed border-orange-400/30 rounded-full animate-spin-reverse"></div>
        
        {/* Hexagonal tech pattern */}
        <div className="absolute top-16 left-1/2 w-40 h-40 transform -translate-x-1/2">
          <div className="w-full h-full border-2 border-amber-500/25 transform rotate-0 animate-pulse" style={{
            clipPath: 'polygon(50% 0%, 100% 25%, 100% 75%, 50% 100%, 0% 75%, 0% 25%)'
          }}></div>
        </div>
        
        {/* Floating rectangles with different rotations */}
        <div className="absolute top-1/4 left-16 w-16 h-32 bg-gradient-to-b from-orange-500/10 to-transparent transform rotate-12 animate-float"></div>
        <div className="absolute top-2/3 right-16 w-12 h-24 bg-gradient-to-t from-amber-500/15 to-transparent transform -rotate-12 animate-float-delay"></div>
        <div className="absolute bottom-1/3 left-1/2 w-8 h-40 bg-gradient-to-b from-orange-400/20 to-transparent transform rotate-45 animate-float-slow"></div>
        
        {/* Circuit-like connections */}
        <div className="absolute top-1/3 left-1/4 w-32 h-2 bg-gradient-to-r from-orange-500/30 to-transparent animate-pulse"></div>
        <div className="absolute top-1/3 left-1/4 w-2 h-32 bg-gradient-to-b from-orange-500/30 to-transparent animate-pulse delay-500"></div>
        <div className="absolute bottom-1/3 right-1/4 w-24 h-2 bg-gradient-to-l from-amber-500/25 to-transparent animate-pulse delay-1000"></div>
        <div className="absolute bottom-1/3 right-1/4 w-2 h-24 bg-gradient-to-t from-amber-500/25 to-transparent animate-pulse delay-1500"></div>
        
        {/* Curved decorative elements */}
        <div className="absolute top-1/4 right-1/3 w-64 h-64 border border-orange-400/15 rounded-full animate-spin-slow"></div>
        <div className="absolute bottom-1/4 left-1/3 w-48 h-48 border-2 border-dashed border-amber-400/20 rounded-full animate-spin-reverse-slow"></div>
        
        {/* Tech grid overlay */}
        <div className="absolute inset-0 opacity-[0.02]" style={{
          backgroundImage: `radial-gradient(circle at 1px 1px, orange 1px, transparent 0)`,
          backgroundSize: '40px 40px'
        }}></div>
        
        {/* Enhanced diagonal pattern */}
        <div className="absolute inset-0 opacity-[0.03]" style={{
          backgroundImage: `repeating-linear-gradient(45deg, transparent, transparent 50px, orange 50px, orange 51px)`
        }}></div>
        
        {/* Additional mesh pattern */}
        <div className="absolute inset-0 opacity-[0.015]" style={{
          backgroundImage: `repeating-linear-gradient(0deg, transparent, transparent 60px, amber 60px, amber 61px), 
                           repeating-linear-gradient(90deg, transparent, transparent 60px, amber 60px, amber 61px)`
        }}></div>
        
        {/* Particle-like dots */}
        <div className="absolute top-1/5 left-1/5 w-2 h-2 bg-orange-400/40 rounded-full animate-ping"></div>
        <div className="absolute top-2/5 right-1/5 w-2 h-2 bg-amber-400/40 rounded-full animate-ping delay-1000"></div>
        <div className="absolute bottom-1/5 left-2/5 w-2 h-2 bg-orange-500/40 rounded-full animate-ping delay-2000"></div>
        <div className="absolute bottom-2/5 right-2/5 w-2 h-2 bg-amber-500/40 rounded-full animate-ping delay-3000"></div>
        
        {/* Glowing accent lines */}
        <div className="absolute top-0 left-1/4 w-1 h-full bg-gradient-to-b from-transparent via-orange-500/10 to-transparent"></div>
        <div className="absolute top-1/4 right-0 w-full h-1 bg-gradient-to-r from-transparent via-amber-500/10 to-transparent"></div>
      </div>

      {/* Navigation */}
      <nav className={`fixed top-0 left-0 right-0 z-50 flex justify-between items-center p-4 sm:p-6 transition-all duration-300 ${
        isScrolled 
          ? 'bg-transparent border-b border-transparent shadow-none' 
          : 'bg-black/80 backdrop-blur-xl border-b border-orange-500/20 shadow-lg'
      }`}>
        <div className="flex items-center">
          <img 
            src="/BOLT logo.png" 
            alt="BOLT Logo" 
            className="h-8 sm:h-10 w-auto object-contain cursor-pointer hover:opacity-80 transition-opacity duration-300"
            onClick={handleLogoClick}
          />
        </div>
        <div className={`flex flex-wrap gap-3 sm:gap-6 transition-all duration-300 ${
          isScrolled ? 'opacity-0 pointer-events-none' : 'opacity-100'
        }`}>
          <button onClick={() => navigateToPage('home')} className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-all duration-300 transform hover:scale-105 ${
            isScrolled 
              ? 'text-orange-400 hover:bg-orange-500/20 border border-transparent' 
              : 'bg-gradient-to-r from-orange-500/20 to-amber-500/20 text-orange-400 border border-orange-500/30 hover:bg-orange-500/30'
          }`}>
            <Home size={20} />
            <span className="font-medium">Home</span>
          </button>
          <button onClick={() => setIsHistoryModalOpen(true)} className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-all duration-300 relative ${
            isScrolled 
              ? 'text-gray-300 hover:text-orange-400 hover:bg-orange-500/20 border border-transparent' 
              : 'text-gray-300 hover:text-orange-400 hover:bg-orange-500/10 border border-transparent hover:border-orange-500/30'
          }`}>
            <History size={20} />
            <span className="font-medium">History</span>
            {allHistory.length > 0 && (
              <span className="absolute -top-2 -right-2 bg-gradient-to-r from-orange-500 to-amber-500 text-white text-xs rounded-full h-5 w-5 flex items-center justify-center shadow-lg animate-bounce">
                {allHistory.length > 99 ? '99+' : allHistory.length}
              </span>
            )}
          </button>
          <button onClick={() => navigateToPage('about')} className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-all duration-300 ${
            isScrolled 
              ? 'text-gray-300 hover:text-orange-400 hover:bg-orange-500/20 border border-transparent' 
              : 'text-gray-300 hover:text-orange-400 hover:bg-orange-500/10 border border-transparent hover:border-orange-500/30'
          }`}>
            <User size={20} />
            <span className="font-medium">About</span>
          </button>
          <button onClick={() => navigateToPage('contact')} className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-all duration-300 ${
            isScrolled 
              ? 'text-gray-300 hover:text-orange-400 hover:bg-orange-500/20 border border-transparent' 
              : 'text-gray-300 hover:text-orange-400 hover:bg-orange-500/10 border border-transparent hover:border-orange-500/30'
          }`}>
            <Mail size={20} />
            <span className="font-medium">Contact</span>
          </button>
        </div>
      </nav>

  {/* Main Content */}
  <div className="relative z-10 container mx-auto px-4 sm:px-6 pt-24 sm:pt-28 py-8 sm:py-12 pb-28 sm:pb-12">
        <div className="text-center mb-12">
          <h1 className="text-3xl sm:text-4xl md:text-5xl lg:text-6xl font-bold text-white mb-4 animate-fade-in">
            {currentInterface === 'lip-reading' ? (
              <>Lip Reading <span className="text-orange-400">Technology</span></>
            ) : (
              <>Hand Gesture <span className="text-orange-400">Technology</span></>
            )}
          </h1>
          <p className="text-base sm:text-lg md:text-xl text-gray-300 max-w-2xl mx-auto">
            {currentInterface === 'lip-reading' 
              ? 'Advanced AI-powered lip reading system that converts visual speech into text with incredible accuracy'
              : 'Real-time hand gesture recognition system powered by advanced computer vision and deep learning'
            }
          </p>
        </div>

        <div className="flex justify-center mb-8">
          <div className="bg-black/40 backdrop-blur-lg rounded-2xl p-2 border border-orange-500/20 inline-flex gap-2">
            <button
              onClick={() => {
                if (currentInterface !== 'lip-reading') {
                  stopHandGestureLive();
                  setCurrentInterface('lip-reading');
                  showToast('Switched to Lip Reading mode', 'info');
                }
              }}
              className={`flex items-center space-x-2 px-6 py-3 rounded-xl font-medium transition-all duration-500 transform ${
                currentInterface === 'lip-reading'
                  ? 'bg-gradient-to-r from-orange-500 to-amber-500 text-white shadow-lg scale-105 border border-orange-400/50'
                  : 'text-gray-300 hover:text-orange-400 hover:bg-orange-500/10 border border-transparent hover:border-orange-500/30'
              }`}
            >
              <Mic size={20} />
              <span>Lip Reading</span>
            </button>
            <button
              onClick={() => {
                if (currentInterface !== 'hand-gesture') {
                  stopLive();
                  setCurrentInterface('hand-gesture');
                  showToast('Switched to Hand Gesture Detection mode', 'info');
                }
              }}
              className={`flex items-center space-x-2 px-6 py-3 rounded-xl font-medium transition-all duration-500 transform ${
                currentInterface === 'hand-gesture'
                  ? 'bg-gradient-to-r from-orange-500 to-amber-500 text-white shadow-lg scale-105 border border-orange-400/50'
                  : 'text-gray-300 hover:text-orange-400 hover:bg-orange-500/10 border border-transparent hover:border-orange-500/30'
              }`}
            >
              <Hand size={20} />
              <span>Gesture Detection</span>
            </button>
          </div>
        </div>

        <div className="max-w-6xl mx-auto">
          {currentInterface === 'lip-reading' ? (
            // Lip Reading Interface
            <div className="grid lg:grid-cols-2 gap-12 items-start">
              {/* Video Section */}
              <div className="space-y-6">
                <div className="relative bg-black/40 backdrop-blur-lg rounded-2xl p-4 sm:p-6 border border-white/10 shadow-2xl">
                  <div className="relative overflow-hidden rounded-xl bg-gray-900">
                    {useServerVideo ? (
                      <img
                        src="http://127.0.0.1:5000/video_feed"
                        alt="Server camera feed"
                        className={`w-full object-cover transition-all duration-300 ${
                          isFullscreen ? 'h-[60vh] md:h-[32rem]' : 'h-56 sm:h-72 md:h-80'
                        }`}
                      />
                    ) : (
                      <video
                        ref={videoRef}
                        autoPlay
                        muted
                        playsInline
                        className={`w-full object-cover transition-all duration-300 ${
                          isFullscreen ? 'h-[60vh] md:h-[32rem]' : 'h-56 sm:h-72 md:h-80'
                        }`}
                      />
                    )}
                    {isLive && (
                      <div className="absolute top-4 right-4">
                        <div className="flex items-center space-x-2 bg-red-500 px-3 py-1 rounded-full">
                          <div className="w-2 h-2 bg-white rounded-full animate-pulse"></div>
                          <span className="text-white text-sm font-medium">Live</span>
                        </div>
                      </div>
                    )}
                    {isProcessing && (
                      <div className="absolute bottom-4 left-4">
                        <div className="bg-blue-500/80 backdrop-blur-sm rounded-lg px-3 py-2 text-center">
                          <div className="flex items-center space-x-2">
                            <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent"></div>
                            <span className="text-white text-sm font-medium">Analyzing...</span>
                          </div>
                        </div>
                      </div>
                    )}
                    <button
                      onClick={toggleFullscreen}
                      className="absolute top-4 left-4 bg-black/50 backdrop-blur-sm rounded-lg p-2 text-white hover:bg-black/70 transition-colors"
                    >
                      {isFullscreen ? <Minimize size={16} /> : <Maximize size={16} />}
                    </button>
                  </div>
                </div>

              {/* Main Controls (desktop/tablet) */}
              <div className="hidden sm:flex flex-col sm:flex-row justify-center gap-3 sm:gap-4">
                <button
                  onClick={isLive ? stopLive : startLive}
                  className={`flex items-center space-x-2 px-8 py-3 rounded-xl font-medium transition-all duration-300 transform hover:scale-105 ${
                    isLive
                      ? 'bg-red-500 hover:bg-red-600 text-white'
                      : 'bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 text-white shadow-lg'
                  }`}
                >
                  {isLive ? <Square size={20} /> : <Camera size={20} />}
                  <span>{isLive ? 'Stop Live' : 'Start Live'}</span>
                </button>

                <button
                  onClick={clearText}
                  className="flex items-center space-x-2 px-6 py-3 bg-gray-600 hover:bg-gray-700 text-white rounded-xl font-medium transition-all duration-300 transform hover:scale-105"
                >
                  <RotateCcw size={20} />
                  <span>Clear</span>
                </button>
              </div>

              {/* Secondary Controls */}
              <div className="hidden sm:grid grid-cols-1 sm:grid-cols-2 gap-3 sm:gap-4">
                <button
                  onClick={saveText}
                  disabled={!transcribedText.trim() || savedTexts.some((it) => it.text === transcribedText)}
                  className="flex items-center justify-center space-x-2 px-4 py-3 bg-green-600 hover:bg-green-700 disabled:bg-gray-500 disabled:cursor-not-allowed text-white rounded-xl font-medium transition-all duration-300 transform hover:scale-105"
                >
                  <Save size={18} />
                  <span>Save Text</span>
                </button>

                <button
                  onClick={toggleAudioSettings}
                  className="flex items-center justify-center space-x-2 px-4 py-3 bg-orange-600 hover:bg-orange-700 text-white rounded-xl font-medium transition-all duration-300 transform hover:scale-105"
                >
                  <Settings size={18} />
                  <span>Audio Settings</span>
                  {isAudioSettingsOpen ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                </button>
              </div>

              {/* Audio Settings Panel - Only opens when clicked */}
              {isAudioSettingsOpen && (
                <div className="bg-black/40 backdrop-blur-lg rounded-2xl p-4 sm:p-6 border border-white/10 shadow-2xl animate-in slide-in-from-top duration-300">
                  <h4 className="text-xl font-bold text-white mb-4 flex items-center">
                    <Volume2 className="mr-3 text-green-400" size={20} />
                    Audio Settings
                    <button onClick={toggleAudioSettings} className="ml-auto text-gray-400 hover:text-white">
                      <ChevronUp size={20} />
                    </button>
                  </h4>
                  <div className="space-y-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-2">Speech Rate: {speechRate.toFixed(1)}x</label>
                      <input
                        type="range"
                        min="0.5"
                        max="2"
                        step="0.1"
                        value={speechRate}
                        onChange={(e) => handleRateChange(e.target.value)}
                        className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer slider"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-2">Pitch: {speechPitch.toFixed(1)}</label>
                      <input
                        type="range"
                        min="0.5"
                        max="2"
                        step="0.1"
                        value={speechPitch}
                        onChange={(e) => handlePitchChange(e.target.value)}
                        className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer slider"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-2">Volume: {Math.round(speechVolume * 100)}%</label>
                      <input
                        type="range"
                        min="0"
                        max="1"
                        step="0.1"
                        value={speechVolume}
                        onChange={(e) => handleVolumeChange(e.target.value)}
                        className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer slider"
                      />
                    </div>
                    {transcribedText && (
                      <button
                        onClick={isPlaying ? stopSpeech : speakText}
                        className={`w-full flex items-center justify-center space-x-2 px-4 py-3 rounded-lg font-medium transition-all duration-300 ${
                          isPlaying ? 'bg-orange-500 hover:bg-orange-600 text-white' : 'bg-green-500 hover:bg-green-600 text-white'
                        }`}
                      >
                        {isPlaying ? <Pause size={18} /> : <Volume2 size={18} />}
                        <span>{isPlaying ? 'Stop Preview' : 'Test Audio'}</span>
                      </button>
                    )}
                  </div>
                </div>
              )}

              {/* Bottom panel under camera: show Recent Saves only when available */}
              {savedTexts.length > 0 && (
                <div className="bg-black/40 backdrop-blur-lg rounded-2xl p-4 sm:p-6 border border-white/10 shadow-2xl">
                  <h4 className="text-xl font-bold text-white mb-4 flex items-center">
                    <History className="mr-3 text-blue-400" size={20} />
                    Recent Saves ({savedTexts.length})
                    <button
                      onClick={unsaveAll}
                      className="ml-auto text-xs px-3 py-1 rounded-md bg-gray-700 hover:bg-gray-600 text-gray-200 border border-white/10"
                    >
                      Unsave All
                    </button>
                  </h4>
                  <div className="space-y-3 max-h-48 sm:max-h-60 overflow-y-auto">
                    {savedTexts.map((item) => {
                      const isExpanded = !!expandedSaved[item.id];
                      const isLong = (item.text?.length || 0) > 200 || (item.text?.split('\n').length || 0) > 3;
                      return (
                        <div key={item.id} className="bg-gray-800/50 rounded-lg p-3">
                          <div className={`relative ${isExpanded ? '' : 'max-h-24 overflow-hidden'}`}>
                            <p className="text-gray-300 text-sm whitespace-pre-wrap">{item.text}</p>
                            {!isExpanded && isLong && (
                              <div className="pointer-events-none absolute bottom-0 left-0 right-0 h-8 bg-gradient-to-t from-gray-900/80 to-transparent"></div>
                            )}
                          </div>
                          {isLong && (
                            <button
                              onClick={() => setExpandedSaved((prev) => ({ ...prev, [item.id]: !isExpanded }))}
                              className="mt-2 text-xs text-purple-300 hover:text-purple-200 underline"
                            >
                              {isExpanded ? 'Show less' : 'Show more'}
                            </button>
                          )}
                          <div className="flex items-center justify-between text-xs text-gray-500 mt-2">
                            <span>{item.timestamp}</span>
                            <span>Confidence: {item.confidence}%</span>
                          </div>
                          <div className="flex items-center space-x-2 mt-3">
                            <button
                              onClick={() => unsaveText(item.id)}
                              className="flex items-center space-x-1 px-3 py-1 bg-orange-600 hover:bg-orange-700 text-white text-xs rounded-md transition-colors"
                            >
                              <BookmarkMinus size={14} />
                              <span>Unsave</span>
                            </button>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>

            {/* Text Output Section */}
            <div className="space-y-6">
              <div className="bg-black/40 backdrop-blur-lg rounded-2xl p-4 sm:p-6 border border-white/10 shadow-2xl">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-2xl font-bold text-white flex items-center">
                    <Mic className="mr-3 text-purple-400" size={24} />
                    Transcribed Text
                  </h3>
                  {transcribedText && (
                    <div className="flex items-center space-x-2">
                      <button
                        onClick={isPlaying ? stopSpeech : speakText}
                        className={`flex items-center space-x-2 px-4 py-2 rounded-lg font-medium transition-all duration-300 transform hover:scale-105 ${
                          isPlaying ? 'bg-orange-500 hover:bg-orange-600 text-white' : 'bg-green-500 hover:bg-green-600 text-white'
                        }`}
                      >
                        {isPlaying ? <Pause size={18} /> : <Volume2 size={18} />}
                        <span className="text-sm">{isPlaying ? 'Stop' : 'Play'}</span>
                      </button>
                      <button
                        onClick={downloadText}
                        className="flex items-center space-x-2 px-4 py-2 rounded-lg font-medium transition-all duration-300 transform hover:scale-105 bg-blue-600 hover:bg-blue-700 text-white"
                      >
                        <Download size={18} />
                        <span className="text-sm">Download</span>
                      </button>
                    </div>
                  )}
                </div>
                <div className="bg-gray-900/50 rounded-xl p-4 sm:p-6 min-h-48 max-h-64 sm:min-h-64 sm:max-h-72 md:min-h-80 md:max-h-80 overflow-y-auto">
                  {transcribedText ? (
                    <div className="space-y-4">
                      <p className="text-gray-300 leading-relaxed text-lg">{transcribedText}</p>
                      <div className="flex items-center justify-between pt-4 border-t border-gray-700">
                        <span className="text-sm text-gray-500">Confidence: {confidence.toFixed(1)}%</span>
                        <span className="text-sm text-gray-500">
                          Words: {transcribedText.split(' ').filter((w) => w.length > 0).length}
                        </span>
                      </div>
                    </div>
                  ) : (
                    <div className="flex items-center justify-center h-full text-gray-500">
                      <div className="text-center">
                        <Camera size={48} className="mx-auto mb-4 opacity-50" />
                        <p className="text-lg">Start live mode to see transcribed text here</p>
                        <p className="text-sm mt-2">Speak clearly while looking at the camera</p>
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {/* Bottom panel under transcribed: Features only when saves exist */}
              {savedTexts.length > 0 && (
                <div className="bg-black/40 backdrop-blur-lg rounded-2xl p-4 sm:p-6 border border-white/10 shadow-2xl">
                  <h4 className="text-xl font-bold text-white mb-4">Features</h4>
                  <div className="space-y-3">
                    <div className="flex items-center space-x-3 text-gray-300">
                      <div className="w-2 h-2 bg-purple-400 rounded-full"></div>
                      <span>Real-time lip movement analysis</span>
                    </div>
                    <div className="flex items-center space-x-3 text-gray-300">
                      <div className="w-2 h-2 bg-purple-400 rounded-full"></div>
                      <span>Text-to-speech audio playback</span>
                    </div>
                    <div className="flex items-center space-x-3 text-gray-300">
                      <div className="w-2 h-2 bg-purple-400 rounded-full"></div>
                      <span>Real-time audio settings adjustment</span>
                    </div>
                    <div className="flex items-center space-x-3 text-gray-300">
                      <div className="w-2 h-2 bg-purple-400 rounded-full"></div>
                      <span>Save and download transcriptions</span>
                    </div>
                    <div className="flex items-center space-x-3 text-gray-300">
                      <div className="w-2 h-2 bg-purple-400 rounded-full"></div>
                      <span>Fullscreen video support</span>
                    </div>
                    <div className="flex items-center space-x-3 text-gray-300">
                      <div className="w-2 h-2 bg-purple-400 rounded-full"></div>
                      <span>Session history tracking</span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
          ) : (
            // Hand Gesture Detection Interface
            <div className="grid lg:grid-cols-2 gap-12 items-start">
              {/* Video Section for Hand Gesture */}
              <div className="space-y-6">
                <div className="relative bg-black/40 backdrop-blur-lg rounded-2xl p-4 sm:p-6 border border-white/10 shadow-2xl">
                  <div className="relative overflow-hidden rounded-xl bg-gray-900">
                    {useServerVideo ? (
                      <img
                        src="http://127.0.0.1:5000/video_feed"
                        alt="Server camera feed"
                        className={`w-full object-cover transition-all duration-300 ${
                          isFullscreen ? 'h-[60vh] md:h-[32rem]' : 'h-56 sm:h-72 md:h-80'
                        }`}
                      />
                    ) : (
                      <video
                        ref={gestureVideoRef}
                        autoPlay
                        muted
                        playsInline
                        className={`w-full object-cover transition-all duration-300 ${
                          isFullscreen ? 'h-[60vh] md:h-[32rem]' : 'h-56 sm:h-72 md:h-80'
                        }`}
                      />
                    )}
                    {isHandGestureLive && (
                      <div className="absolute top-4 right-4">
                        <div className="flex items-center space-x-2 bg-purple-500 px-3 py-1 rounded-full">
                          <div className="w-2 h-2 bg-white rounded-full animate-pulse"></div>
                          <span className="text-white text-sm font-medium">Detecting</span>
                        </div>
                      </div>
                    )}
                    {isHandGestureDemo && (
                      <div className="absolute top-4 right-4">
                        <div className="flex items-center space-x-2 bg-pink-500 px-3 py-1 rounded-full">
                          <Play size={14} className="text-white" />
                          <span className="text-white text-sm font-medium">Demo</span>
                        </div>
                      </div>
                    )}
                    {isHandGestureProcessing && (
                      <div className="absolute bottom-4 left-4">
                        <div className="bg-blue-500/80 backdrop-blur-sm rounded-lg px-3 py-2 text-center">
                          <div className="flex items-center space-x-2">
                            <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent"></div>
                            <span className="text-white text-sm font-medium">Processing...</span>
                          </div>
                        </div>
                      </div>
                    )}
                    <button
                      onClick={toggleFullscreen}
                      className="absolute top-4 left-4 bg-black/50 backdrop-blur-sm rounded-lg p-2 text-white hover:bg-black/70 transition-colors"
                    >
                      {isFullscreen ? <Minimize size={16} /> : <Maximize size={16} />}
                    </button>
                    {/* Demo close button */}
                    {isHandGestureDemo && (
                      <button
                        onClick={closeDemo}
                        className="absolute top-4 right-16 bg-red-500/80 backdrop-blur-sm rounded-lg p-2 text-white hover:bg-red-600/80 transition-colors"
                      >
                        <X size={16} />
                      </button>
                    )}
                  </div>
                </div>

                {/* Hand Gesture Controls */}
                <div className="hidden sm:flex flex-col sm:flex-row justify-center gap-3 sm:gap-4">
                  <button
                    onClick={isHandGestureLive ? stopHandGestureLive : startHandGestureLive}
                    className={`flex items-center space-x-2 px-8 py-3 rounded-xl font-medium transition-all duration-300 transform hover:scale-105 ${
                      isHandGestureLive
                        ? 'bg-red-500 hover:bg-red-600 text-white'
                        : 'bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 text-white shadow-lg'
                    }`}
                  >
                    {isHandGestureLive ? <Square size={20} /> : <Hand size={20} />}
                    <span>{isHandGestureLive ? 'Stop Detection' : 'Start Detection'}</span>
                  </button>

                  <button
                    onClick={clearGestureText}
                    className="flex items-center space-x-2 px-6 py-3 bg-gray-600 hover:bg-gray-700 text-white rounded-xl font-medium transition-all duration-300 transform hover:scale-105"
                  >
                    <RotateCcw size={20} />
                    <span>Clear</span>
                  </button>
                </div>

                {/* Secondary Controls */}
                <div className="hidden sm:grid grid-cols-1 sm:grid-cols-2 gap-3 sm:gap-4">
                  <button
                    onClick={saveGestureText}
                    disabled={!gestureText.trim() || savedGestures.some((it) => it.text === gestureText)}
                    className="flex items-center justify-center space-x-2 px-4 py-3 bg-green-600 hover:bg-green-700 disabled:bg-gray-500 disabled:cursor-not-allowed text-white rounded-xl font-medium transition-all duration-300 transform hover:scale-105"
                  >
                    <Save size={18} />
                    <span>Save Text</span>
                  </button>

                  <button
                    onClick={toggleGestureAudioSettings}
                    className="flex items-center justify-center space-x-2 px-4 py-3 bg-orange-600 hover:bg-orange-700 text-white rounded-xl font-medium transition-all duration-300 transform hover:scale-105"
                  >
                    <Settings size={18} />
                    <span>Audio Settings</span>
                    {isGestureAudioSettingsOpen ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                  </button>
                </div>

                {/* Gesture Audio Settings Panel - Only opens when clicked */}
                {isGestureAudioSettingsOpen && (
                  <div className="bg-black/40 backdrop-blur-lg rounded-2xl p-4 sm:p-6 border border-white/10 shadow-2xl animate-in slide-in-from-top duration-300">
                    <h4 className="text-xl font-bold text-white mb-4 flex items-center">
                      <Volume2 className="mr-3 text-purple-400" size={20} />
                      Audio Settings
                      <button onClick={toggleGestureAudioSettings} className="ml-auto text-gray-400 hover:text-white">
                        <ChevronUp size={20} />
                      </button>
                    </h4>
                    <div className="space-y-4">
                      <div>
                        <label className="block text-sm font-medium text-gray-300 mb-2">Speech Rate: {speechRate.toFixed(1)}x</label>
                        <input
                          type="range"
                          min="0.5"
                          max="2"
                          step="0.1"
                          value={speechRate}
                          onChange={(e) => handleRateChange(e.target.value)}
                          className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer slider"
                        />
                      </div>
                      <div>
                        <label className="block text-sm font-medium text-gray-300 mb-2">Pitch: {speechPitch.toFixed(1)}</label>
                        <input
                          type="range"
                          min="0.5"
                          max="2"
                          step="0.1"
                          value={speechPitch}
                          onChange={(e) => handlePitchChange(e.target.value)}
                          className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer slider"
                        />
                      </div>
                      <div>
                        <label className="block text-sm font-medium text-gray-300 mb-2">Volume: {Math.round(speechVolume * 100)}%</label>
                        <input
                          type="range"
                          min="0"
                          max="1"
                          step="0.1"
                          value={speechVolume}
                          onChange={(e) => handleVolumeChange(e.target.value)}
                          className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer slider"
                        />
                      </div>
                      {gestureText && (
                        <button
                          onClick={isPlaying ? stopSpeech : speakGestureText}
                          className={`w-full flex items-center justify-center space-x-2 px-4 py-3 rounded-lg font-medium transition-all duration-300 ${
                            isPlaying ? 'bg-orange-500 hover:bg-orange-600 text-white' : 'bg-green-500 hover:bg-green-600 text-white'
                          }`}
                        >
                          {isPlaying ? <Pause size={18} /> : <Volume2 size={18} />}
                          <span>{isPlaying ? 'Stop Preview' : 'Test Audio'}</span>
                        </button>
                      )}
                    </div>
                  </div>
                )}

                {/* Bottom panel under camera: show Recent Saves only when available */}
                {savedGestures.length > 0 && (
                  <div className="bg-black/40 backdrop-blur-lg rounded-2xl p-4 sm:p-6 border border-white/10 shadow-2xl">
                    <h4 className="text-xl font-bold text-white mb-4 flex items-center">
                      <History className="mr-3 text-purple-400" size={20} />
                      Recent Saves ({savedGestures.length})
                      <button
                        onClick={unsaveAllGestures}
                        className="ml-auto text-xs px-3 py-1 rounded-md bg-gray-700 hover:bg-gray-600 text-gray-200 border border-white/10"
                      >
                        Unsave All
                      </button>
                    </h4>
                    <div className="space-y-3 max-h-48 sm:max-h-60 overflow-y-auto">
                      {savedGestures.map((item) => (
                        <div key={item.id} className="bg-gray-800/50 rounded-lg p-3">
                          <p className="text-gray-300 text-sm whitespace-pre-wrap">{item.text}</p>
                          <div className="flex items-center justify-between text-xs text-gray-500 mt-2">
                            <span>{item.timestamp}</span>
                            <span>Confidence: {item.confidence}%</span>
                          </div>
                          <div className="flex items-center space-x-2 mt-3">
                            <button
                              onClick={() => unsaveGestureText(item.id)}
                              className="flex items-center space-x-1 px-3 py-1 bg-orange-600 hover:bg-orange-700 text-white text-xs rounded-md transition-colors"
                            >
                              <BookmarkMinus size={14} />
                              <span>Unsave</span>
                            </button>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              {/* Hand Gesture Text Output Section */}
              <div className="space-y-6">
                <div className="bg-black/40 backdrop-blur-lg rounded-2xl p-4 sm:p-6 border border-white/10 shadow-2xl">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-2xl font-bold text-white flex items-center">
                      <Hand className="mr-3 text-purple-400" size={24} />
                      Gesture Text
                    </h3>
                    <div className="flex items-center space-x-2">
                      <button
                        onClick={startDemo}
                        disabled={isHandGestureDemo || isHandGestureLive}
                        className="flex items-center space-x-2 px-4 py-2 rounded-lg font-medium transition-all duration-300 transform hover:scale-105 bg-pink-600 hover:bg-pink-700 disabled:bg-gray-500 disabled:cursor-not-allowed text-white"
                      >
                        <Play size={18} />
                        <span className="text-sm">Demo</span>
                      </button>
                      {gestureText && (
                        <>
                          <button
                            onClick={isPlaying ? stopSpeech : speakGestureText}
                            className={`flex items-center space-x-2 px-4 py-2 rounded-lg font-medium transition-all duration-300 transform hover:scale-105 ${
                              isPlaying ? 'bg-orange-500 hover:bg-orange-600 text-white' : 'bg-green-500 hover:bg-green-600 text-white'
                            }`}
                          >
                            {isPlaying ? <Pause size={18} /> : <Volume2 size={18} />}
                            <span className="text-sm">{isPlaying ? 'Stop' : 'Play'}</span>
                          </button>
                          <button
                            onClick={downloadGestureText}
                            className="flex items-center space-x-2 px-4 py-2 rounded-lg font-medium transition-all duration-300 transform hover:scale-105 bg-blue-600 hover:bg-blue-700 text-white"
                          >
                            <Download size={18} />
                            <span className="text-sm">Download</span>
                          </button>
                        </>
                      )}
                    </div>
                  </div>
                  <div className="bg-gray-900/50 rounded-xl p-6 sm:p-8 min-h-48 max-h-64 sm:min-h-64 sm:max-h-72 md:min-h-80 md:max-h-80 overflow-y-auto">
                    {gestureText ? (
                      <div className="space-y-6">
                        <div className="space-y-4">
                          <p className="text-gray-300 leading-relaxed text-lg">{gestureText}</p>
                          <div className="flex items-center justify-between pt-4 border-t border-gray-700">
                            <span className="text-sm text-gray-500">Confidence: {gestureConfidence.toFixed(1)}%</span>
                            <span className="text-sm text-gray-500">Total Detected: {savedGestures.length}</span>
                          </div>
                        </div>
                      </div>
                    ) : (
                      <div className="flex items-center justify-center h-full text-gray-500">
                        <div className="text-center">
                          <Hand size={64} className="mx-auto mb-4 opacity-50 text-purple-400" />
                          <p className="text-lg text-gray-300">Start detection to see gesture interpretations here</p>
                          <p className="text-sm mt-2 text-gray-400">Show your hand gesture to the camera</p>
                        </div>
                      </div>
                    )}
                  </div>
                </div>

                {/* Bottom panel under gesture text: Features only when saves exist */}
                {savedGestures.length > 0 && (
                  <div className="bg-black/40 backdrop-blur-lg rounded-2xl p-4 sm:p-6 border border-white/10 shadow-2xl">
                    <h4 className="text-xl font-bold text-white mb-4">Features</h4>
                    <div className="space-y-3">
                      <div className="flex items-center space-x-3 text-gray-300">
                        <div className="w-2 h-2 bg-purple-400 rounded-full"></div>
                        <span>Real-time hand gesture recognition</span>
                      </div>
                      <div className="flex items-center space-x-3 text-gray-300">
                        <div className="w-2 h-2 bg-purple-400 rounded-full"></div>
                        <span>Multiple gesture types supported</span>
                      </div>
                      <div className="flex items-center space-x-3 text-gray-300">
                        <div className="w-2 h-2 bg-purple-400 rounded-full"></div>
                        <span>Text-to-speech for gestures</span>
                      </div>
                      <div className="flex items-center space-x-3 text-gray-300">
                        <div className="w-2 h-2 bg-purple-400 rounded-full"></div>
                        <span>Save and download gesture data</span>
                      </div>
                      <div className="flex items-center space-x-3 text-gray-300">
                        <div className="w-2 h-2 bg-purple-400 rounded-full"></div>
                        <span>Fullscreen video support</span>
                      </div>
                      <div className="flex items-center space-x-3 text-gray-300">
                        <div className="w-2 h-2 bg-purple-400 rounded-full"></div>
                        <span>Gesture history tracking</span>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
        
        {/* Full-width Features: shown only when no saves exist for lip-reading interface */}
        {currentInterface === 'lip-reading' && savedTexts.length === 0 && (
          <div className="mt-8 sm:mt-12">
            <div className="bg-black/40 backdrop-blur-lg rounded-2xl p-6 sm:p-8 border border-white/10 shadow-2xl">
              <h4 className="text-2xl font-bold text-white mb-6">Features</h4>
              <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4 text-gray-300">
                <div className="flex items-center space-x-3">
                  <div className="w-2 h-2 bg-purple-400 rounded-full"></div>
                  <span>Real-time lip movement analysis</span>
                </div>
                <div className="flex items-center space-x-3">
                  <div className="w-2 h-2 bg-purple-400 rounded-full"></div>
                  <span>Text-to-speech audio playback</span>
                </div>
                <div className="flex items-center space-x-3">
                  <div className="w-2 h-2 bg-purple-400 rounded-full"></div>
                  <span>Real-time audio settings adjustment</span>
                </div>
                <div className="flex items-center space-x-3">
                  <div className="w-2 h-2 bg-purple-400 rounded-full"></div>
                  <span>Save and download transcriptions</span>
                </div>
                <div className="flex items-center space-x-3">
                  <div className="w-2 h-2 bg-purple-400 rounded-full"></div>
                  <span>Fullscreen video support</span>
                </div>
                <div className="flex items-center space-x-3">
                  <div className="w-2 h-2 bg-purple-400 rounded-full"></div>
                  <span>Session history tracking</span>
                </div>
              </div>
            </div>
          </div>
        )}
        
        {/* Full-width Features: shown only when no saves exist for hand gesture interface */}
        {currentInterface === 'hand-gesture' && savedGestures.length === 0 && (
          <div className="mt-8 sm:mt-12">
            <div className="bg-black/40 backdrop-blur-lg rounded-2xl p-6 sm:p-8 border border-white/10 shadow-2xl">
              <h4 className="text-2xl font-bold text-white mb-6">Features</h4>
              <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-6 text-gray-300">
                <div className="flex items-center space-x-3">
                  <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                  <span>Real-time hand gesture recognition</span>
                </div>
                <div className="flex items-center space-x-3">
                  <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                  <span>Multiple gesture types supported</span>
                </div>
                <div className="flex items-center space-x-3">
                  <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                  <span>Text-to-speech for gestures</span>
                </div>
                <div className="flex items-center space-x-3">
                  <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                  <span>Save and download gesture data</span>
                </div>
                <div className="flex items-center space-x-3">
                  <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                  <span>Fullscreen video support</span>
                </div>
                <div className="flex items-center space-x-3">
                  <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                  <span>Gesture history tracking</span>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Mobile sticky action bar */}
      <div className="sm:hidden fixed bottom-0 left-0 right-0 z-40 bg-black/50 backdrop-blur-lg border-t border-white/10 px-4 py-3 flex items-center justify-center gap-3" style={{ paddingBottom: 'calc(env(safe-area-inset-bottom, 0px) + 0.5rem)' }}>
        {currentInterface === 'lip-reading' ? (
          // Lip Reading Mobile Controls
          <>
            <button
              onClick={isLive ? stopLive : startLive}
              className={`flex-1 flex items-center justify-center space-x-2 px-4 py-3 rounded-xl font-medium transition-all duration-300 ${
                isLive ? 'bg-red-500 text-white' : 'bg-gradient-to-r from-purple-500 to-pink-500 text-white'
              }`}
            >
              {isLive ? <Square size={18} /> : <Camera size={18} />}
              <span>{isLive ? 'Stop' : 'Start'}</span>
            </button>
            <button
              onClick={saveText}
              disabled={!transcribedText.trim() || savedTexts.some((it) => it.text === transcribedText)}
              className="flex-1 flex items-center justify-center space-x-2 px-4 py-3 bg-green-600 disabled:bg-gray-500 disabled:cursor-not-allowed text-white rounded-xl font-medium"
            >
              <Save size={18} />
              <span>Save</span>
            </button>
            <button
              onClick={toggleAudioSettings}
              className="flex items-center justify-center p-3 bg-orange-600 text-white rounded-xl"
            >
              <Settings size={18} />
            </button>
          </>
        ) : (
          // Hand Gesture Mobile Controls
          <>
            <button
              onClick={isHandGestureLive ? stopHandGestureLive : startHandGestureLive}
              className={`flex-1 flex items-center justify-center space-x-2 px-4 py-3 rounded-xl font-medium transition-all duration-300 ${
                isHandGestureLive ? 'bg-red-500 text-white' : 'bg-gradient-to-r from-emerald-500 to-cyan-500 text-white'
              }`}
            >
              {isHandGestureLive ? <Square size={18} /> : <Hand size={18} />}
              <span>{isHandGestureLive ? 'Stop' : 'Detect'}</span>
            </button>
            <button
              onClick={saveGestureText}
              disabled={!gestureText.trim() || savedGestures.some((it) => it.text === gestureText)}
              className="flex-1 flex items-center justify-center space-x-2 px-4 py-3 bg-green-600 disabled:bg-gray-500 disabled:cursor-not-allowed text-white rounded-xl font-medium"
            >
              <Save size={18} />
              <span>Save</span>
            </button>
            <button
              onClick={toggleGestureAudioSettings}
              className="flex items-center justify-center p-3 bg-orange-600 text-white rounded-xl"
            >
              <Settings size={18} />
            </button>
          </>
        )}
      </div>
    </div>
  );

  const renderAbout = () => (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-black to-slate-900 relative overflow-hidden transition-all duration-700 ease-in-out transform">
      {/* Professional background graphics */}
      <div className="absolute inset-0">
        {/* Animated gradient orbs */}
        <div className="absolute top-32 left-32 w-80 h-80 bg-gradient-to-r from-orange-500/25 to-amber-500/25 rounded-full blur-3xl animate-pulse delay-500"></div>
        <div className="absolute bottom-32 right-32 w-96 h-96 bg-gradient-to-r from-amber-400/20 to-yellow-500/20 rounded-full blur-3xl animate-pulse delay-1500"></div>
        
        {/* Tech-inspired patterns */}
        <div className="absolute top-20 right-20 w-32 h-32 border-2 border-orange-400/30 rounded-lg transform rotate-12 animate-slow-bounce"></div>
        <div className="absolute bottom-40 left-20 w-24 h-24 border border-dashed border-orange-500/40 rotate-45"></div>
        
        {/* Circuit-like lines */}
        <div className="absolute top-1/4 left-0 w-full h-px bg-gradient-to-r from-transparent via-orange-500/30 to-transparent"></div>
        <div className="absolute bottom-1/4 left-0 w-full h-px bg-gradient-to-r from-transparent via-amber-500/30 to-transparent"></div>
        
        {/* Hexagonal pattern */}
        <div className="absolute inset-0 opacity-[0.02]" style={{
          backgroundImage: `url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23f97316' fill-opacity='1'%3E%3Cpath d='M30 60L45 45v-30L30 0L15 15v30L30 60z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E")`,
          backgroundSize: '60px 60px'
        }}></div>
      </div>

      <nav className={`fixed top-0 left-0 right-0 z-50 flex justify-between items-center p-4 sm:p-6 transition-all duration-300 ${
        isScrolled 
          ? 'bg-transparent border-b border-transparent shadow-none' 
          : 'bg-black/80 backdrop-blur-xl border-b border-orange-500/20 shadow-lg'
      }`}>
        <div className="flex items-center">
          <button onClick={() => navigateToPage('home')} className={`flex items-center space-x-2 px-2 py-2 rounded-lg text-gray-300 hover:text-orange-400 hover:bg-orange-500/10 border border-transparent hover:border-orange-500/30 transition-all duration-300 ${
            isScrolled ? 'opacity-0 pointer-events-none' : 'opacity-100'
          }`}>
            <ArrowLeft size={20} />
          </button>
          <img 
            src="/BOLT logo.png" 
            alt="BOLT Logo" 
            className="h-8 sm:h-10 w-auto object-contain ml-2 cursor-pointer hover:opacity-80 transition-opacity duration-300"
            onClick={handleLogoClick}
          />
        </div>
        <div className={`flex flex-wrap gap-3 sm:gap-6 transition-all duration-300 ${
          isScrolled ? 'opacity-0 pointer-events-none' : 'opacity-100'
        }`}>
          <button onClick={() => navigateToPage('home')} className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-all duration-300 ${
            isScrolled 
              ? 'text-gray-300 hover:text-orange-400 hover:bg-orange-500/20 border border-transparent' 
              : 'text-gray-300 hover:text-orange-400 hover:bg-orange-500/10 border border-transparent hover:border-orange-500/30'
          }`}>
            <Home size={20} />
            <span className="font-medium">Home</span>
          </button>
          <button onClick={() => navigateToPage('about')} className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-all duration-300 ${
            isScrolled 
              ? 'bg-gradient-to-r from-orange-500/20 to-amber-500/20 text-orange-400 border border-transparent' 
              : 'bg-gradient-to-r from-orange-500/20 to-amber-500/20 text-orange-400 border border-orange-500/30'
          }`}>
            <User size={20} />
            <span className="font-medium">About</span>
          </button>
          <button onClick={() => navigateToPage('contact')} className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-all duration-300 ${
            isScrolled 
              ? 'text-gray-300 hover:text-orange-400 hover:bg-orange-500/20 border border-transparent' 
              : 'text-gray-300 hover:text-orange-400 hover:bg-orange-500/10 border border-transparent hover:border-orange-500/30'
          }`}>
            <Mail size={20} />
            <span className="font-medium">Contact</span>
          </button>
        </div>
      </nav>

      <div className="relative z-10 container mx-auto px-6 pt-24 py-16">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-6xl font-bold text-white mb-8 animate-fade-in">
            About <span className="bg-gradient-to-r from-orange-400 via-amber-500 to-orange-600 bg-clip-text text-transparent">BOLT</span>
          </h1>

          <div className="bg-gradient-to-br from-gray-900/80 via-black/60 to-gray-800/80 backdrop-blur-xl rounded-3xl p-8 border border-orange-500/20 shadow-2xl relative overflow-hidden">
            {/* Card decoration */}
            <div className="absolute top-0 right-0 w-32 h-32 bg-gradient-to-br from-orange-500/10 to-transparent rounded-bl-3xl"></div>
            <div className="absolute bottom-0 left-0 w-32 h-32 bg-gradient-to-tr from-amber-500/10 to-transparent rounded-tr-3xl"></div>
            
            <p className="text-xl text-gray-300 leading-relaxed mb-6 relative z-10">
              BOLT represents the cutting-edge fusion of artificial intelligence and computer vision technology, designed to revolutionize communication accessibility through advanced lip reading capabilities.
            </p>

            <div className="grid md:grid-cols-2 gap-8 mt-8 relative z-10">
              <div className="space-y-6 p-6 bg-gradient-to-br from-orange-500/5 to-amber-500/5 rounded-2xl border border-orange-500/10">
                <h3 className="text-2xl font-bold text-white flex items-center gap-3">
                  <div className="w-8 h-8 bg-gradient-to-r from-orange-500 to-amber-500 rounded-lg flex items-center justify-center">
                    <span className="text-white font-bold text-sm">M</span>
                  </div>
                  Our Mission
                </h3>
                <p className="text-gray-300">
                  To break down communication barriers and provide seamless, accurate lip reading technology that empowers individuals with hearing impairments and enhances communication in noisy environments.
                </p>
              </div>

              <div className="space-y-6 p-6 bg-gradient-to-br from-amber-500/5 to-orange-500/5 rounded-2xl border border-amber-500/10">
                <h3 className="text-2xl font-bold text-white flex items-center gap-3">
                  <div className="w-8 h-8 bg-gradient-to-r from-amber-500 to-orange-500 rounded-lg flex items-center justify-center">
                    <span className="text-white font-bold text-sm">T</span>
                  </div>
                  Technology
                </h3>
                <p className="text-gray-300">
                  Powered by state-of-the-art neural networks and real-time computer vision algorithms, BOLT analyzes lip movements with unprecedented accuracy and speed, now featuring integrated text-to-speech capabilities with real-time adjustments.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  const renderContact = () => (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-black to-gray-900 relative overflow-hidden transition-all duration-700 ease-in-out transform">
      {/* Professional background graphics */}
      <div className="absolute inset-0">
        {/* Large gradient orbs */}
        <div className="absolute top-40 right-40 w-96 h-96 bg-gradient-to-r from-orange-500/20 to-amber-600/20 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute bottom-40 left-40 w-80 h-80 bg-gradient-to-r from-amber-500/15 to-yellow-500/15 rounded-full blur-3xl animate-pulse delay-1000"></div>
        
        {/* Modern geometric shapes */}
        <div className="absolute top-20 left-20 w-40 h-40 border-2 border-orange-400/25 rounded-2xl transform rotate-12 animate-slow-spin"></div>
        <div className="absolute bottom-20 right-20 w-32 h-32 bg-gradient-to-br from-orange-500/10 to-transparent rounded-full"></div>
        
        {/* Contact-themed decorative elements */}
        <div className="absolute top-1/3 right-10 w-2 h-32 bg-gradient-to-b from-orange-500/40 to-transparent"></div>
        <div className="absolute bottom-1/3 left-10 w-2 h-24 bg-gradient-to-t from-amber-500/40 to-transparent"></div>
        
        {/* Dot pattern overlay */}
        <div className="absolute inset-0 opacity-[0.03]" style={{
          backgroundImage: `radial-gradient(circle at 2px 2px, orange 1px, transparent 0)`,
          backgroundSize: '30px 30px'
        }}></div>
      </div>

      <nav className={`fixed top-0 left-0 right-0 z-50 flex justify-between items-center p-4 sm:p-6 transition-all duration-300 ${
        isScrolled 
          ? 'bg-transparent border-b border-transparent shadow-none' 
          : 'bg-black/80 backdrop-blur-xl border-b border-orange-500/20 shadow-lg'
      }`}>
        <div className="flex items-center">
          <button onClick={() => navigateToPage('home')} className={`flex items-center space-x-2 px-2 py-2 rounded-lg text-gray-300 hover:text-orange-400 hover:bg-orange-500/10 border border-transparent hover:border-orange-500/30 transition-all duration-300 ${
            isScrolled ? 'opacity-0 pointer-events-none' : 'opacity-100'
          }`}>
            <ArrowLeft size={20} />
          </button>
          <img 
            src="/BOLT logo.png" 
            alt="BOLT Logo" 
            className="h-8 sm:h-10 w-auto object-contain ml-2 cursor-pointer hover:opacity-80 transition-opacity duration-300"
            onClick={handleLogoClick}
          />
        </div>
        <div className={`flex flex-wrap gap-3 sm:gap-6 transition-all duration-300 ${
          isScrolled ? 'opacity-0 pointer-events-none' : 'opacity-100'
        }`}>
          <button onClick={() => navigateToPage('home')} className="flex items-center space-x-2 px-4 py-2 rounded-lg text-gray-300 hover:text-orange-400 hover:bg-orange-500/10 border border-transparent hover:border-orange-500/30 transition-all duration-300">
            <Home size={20} />
            <span className="font-medium">Home</span>
          </button>
          <button onClick={() => navigateToPage('about')} className="flex items-center space-x-2 px-4 py-2 rounded-lg text-gray-300 hover:text-orange-400 hover:bg-orange-500/10 border border-transparent hover:border-orange-500/30 transition-all duration-300">
            <User size={20} />
            <span className="font-medium">About</span>
          </button>
          <button onClick={() => navigateToPage('contact')} className="flex items-center space-x-2 px-4 py-2 rounded-lg bg-gradient-to-r from-orange-500/20 to-amber-500/20 text-orange-400 border border-orange-500/30">
            <Mail size={20} />
            <span className="font-medium">Contact</span>
          </button>
        </div>
      </nav>

      <div className="relative z-10 container mx-auto px-6 pt-24 py-16">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-6xl font-bold text-white mb-8 animate-fade-in">
            Contact <span className="text-orange-400">Us</span>
          </h1>

          <div className="bg-gray-900/60 backdrop-blur-lg rounded-2xl p-8 border border-gray-700 shadow-2xl">
            <div className="grid md:grid-cols-2 gap-8">
              <div className="space-y-6">
                <h3 className="text-2xl font-bold text-white">Get in Touch</h3>
                <p className="text-gray-300">
                  Have questions about BOLT (Breaking Obstacles with Lip-reading Technology) or want to learn more about our assistive communication system? We'd love to hear from you.
                </p>

                <div className="space-y-4">
                  <div className="flex items-center space-x-3 text-gray-300">
                    <Mail className="text-orange-400" size={20} />
                    <span>bolt@sjbit.edu.in</span>
                  </div>
                  <div className="flex items-center space-x-3 text-gray-300">
                    <User className="text-orange-400" size={20} />
                    <span>SJB Institute of Technology Team</span>
                  </div>
                  <div className="flex items-center space-x-3 text-gray-300">
                    <Camera className="text-orange-400" size={20} />
                    <span>VTU Project 2024-25</span>
                  </div>
                </div>

                <div className="mt-6 p-4 bg-gray-800/50 rounded-lg border border-gray-700">
                  <h4 className="text-lg font-semibold text-white mb-2">Project Team</h4>
                  <div className="text-sm text-gray-300 space-y-1">
                    <p>Sudeeksha T [1JB22IS157]</p>
                    <p>T J Shashank [1JB22IS164]</p>
                    <p>Tanusha Urs M [1JB22IS165]</p>
                    <p>Veeresh H P [1JB22IS180]</p>
                    <p className="mt-2 text-orange-400">Under guidance of Prof. Gayathri G</p>
                  </div>
                </div>
              </div>

              <div className="space-y-4">
                <input
                  type="text"
                  placeholder="Your Name"
                  className="w-full px-4 py-3 bg-gray-800/50 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-orange-400 transition-colors"
                />
                <input
                  type="email"
                  placeholder="Your Email"
                  className="w-full px-4 py-3 bg-gray-800/50 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-orange-400 transition-colors"
                />
                <textarea
                  rows="4"
                  placeholder="Your Message"
                  className="w-full px-4 py-3 bg-gray-800/50 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-orange-400 transition-colors resize-none"
                ></textarea>
                <button className="w-full bg-gradient-to-r from-orange-500 to-orange-600 hover:from-orange-600 hover:to-orange-700 text-white py-3 rounded-lg font-medium transition-all duration-300 transform hover:scale-105">
                  Send Message
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  return (
    <div className="font-sans">
      <style>{`
        .slider::-webkit-slider-thumb {
          appearance: none;
          height: 20px;
          width: 20px;
          border-radius: 50%;
          background: #8b5cf6;
          cursor: pointer;
        }
        .slider::-moz-range-thumb {
          height: 20px;
          width: 20px;
          border-radius: 50%;
          background: #8b5cf6;
          cursor: pointer;
          border: none;
        }
        
        @keyframes slow-spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
        
        @keyframes slow-bounce {
          0%, 100% { transform: translateY(0) rotate(12deg); }
          50% { transform: translateY(-10px) rotate(12deg); }
        }
        
        .animate-slow-spin {
          animation: slow-spin 20s linear infinite;
        }
        
        .animate-slow-bounce {
          animation: slow-bounce 3s ease-in-out infinite;
        }
        
        @keyframes fade-in {
          from { opacity: 0; transform: translateY(30px); }
          to { opacity: 1; transform: translateY(0); }
        }
        
        .animate-fade-in {
          animation: fade-in 1s ease-out;
        }
      `}</style>
      {currentPage === 'home' && renderHome()}
      {currentPage === 'about' && renderAbout()}
      {currentPage === 'contact' && renderContact()}

      {/* Toasts */}
      <div
        className={`fixed z-50 space-y-2 ${
          TOAST_POSITION === 'bottom-right'
            ? 'right-4 bottom-20 sm:bottom-4'
            : TOAST_POSITION === 'bottom-left'
            ? 'left-4 bottom-20 sm:bottom-4'
            : TOAST_POSITION === 'top-right'
            ? 'top-4 right-4'
            : 'top-4 left-4'
        }`}
      >
        {toasts.map((t) => {
          const base = TOAST_THEME === 'light' ? 'bg-white text-gray-900 border-gray-300' : 'text-white border-orange-500/20';
          const color =
            t.type === 'success'
              ? TOAST_THEME === 'light'
                ? 'bg-green-100 border-green-300 text-green-900'
                : 'bg-gradient-to-r from-green-600/90 to-emerald-600/90 border-green-400/40'
              : t.type === 'info'
              ? TOAST_THEME === 'light'
                ? 'bg-orange-100 border-orange-300 text-orange-900'
                : 'bg-gradient-to-r from-orange-600/90 to-amber-600/90 border-orange-400/40'
              : TOAST_THEME === 'light'
              ? 'bg-gray-100 border-gray-300 text-gray-900'
              : 'bg-gradient-to-r from-gray-700/90 to-gray-600/90 border-orange-500/20';
          const Icon = t.type === 'success' ? CheckCircle2 : t.type === 'info' ? InfoIcon : AlertTriangle;
          return (
            <div key={t.id} className={`px-4 py-2 rounded-lg shadow-lg backdrop-blur-sm flex items-center space-x-2 ${base} ${color}`}>
              <Icon size={18} />
              <span className="text-sm font-medium">{t.message}</span>
            </div>
          );
        })}
      </div>

      {/* History Modal */}
      {isHistoryModalOpen && (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-gradient-to-br from-slate-900 via-black to-gray-900 rounded-2xl border border-orange-500/20 w-full max-w-4xl max-h-[90vh] shadow-2xl flex flex-col relative overflow-hidden">
            {/* Professional background graphics */}
            <div className="absolute inset-0">
              {/* Large gradient orbs */}
              <div className="absolute top-20 right-20 w-64 h-64 bg-gradient-to-r from-orange-500/20 to-amber-600/20 rounded-full blur-3xl animate-pulse"></div>
              <div className="absolute bottom-20 left-20 w-56 h-56 bg-gradient-to-r from-amber-500/15 to-yellow-500/15 rounded-full blur-3xl animate-pulse delay-1000"></div>
              
              {/* Modern geometric shapes */}
              <div className="absolute top-16 left-16 w-32 h-32 border-2 border-orange-400/25 rounded-2xl transform rotate-12 animate-slow-spin"></div>
              <div className="absolute bottom-16 right-16 w-28 h-28 bg-gradient-to-br from-orange-500/10 to-transparent rounded-full"></div>
              
              {/* History-themed decorative elements */}
              <div className="absolute top-1/3 right-8 w-2 h-24 bg-gradient-to-b from-orange-500/40 to-transparent"></div>
              <div className="absolute bottom-1/3 left-8 w-2 h-20 bg-gradient-to-t from-amber-500/40 to-transparent"></div>
              
              {/* Dot pattern overlay */}
              <div className="absolute inset-0 opacity-[0.03]" style={{
                backgroundImage: `radial-gradient(circle at 2px 2px, orange 1px, transparent 0)`,
                backgroundSize: '25px 25px'
              }}></div>
            </div>

            {/* Modal Header with Tabs */}
            <div className="relative z-10 flex items-center justify-between p-6 border-b border-orange-500/20 bg-black/40 backdrop-blur-lg">
              <h2 className="text-2xl font-bold text-white flex items-center">
                <History className="mr-3 text-orange-400" size={24} />
                <span className="bg-gradient-to-r from-orange-400 via-amber-500 to-orange-600 bg-clip-text text-transparent">
                  History
                </span>
              </h2>
              <button onClick={() => setIsHistoryModalOpen(false)} className="text-gray-400 hover:text-orange-400 transition-all duration-300 p-2 hover:bg-orange-500/10 rounded-lg border border-transparent hover:border-orange-500/30">
                <X size={20} />
              </button>
            </div>


            {/* Modal Content */}
            <div className="relative z-10 p-6 overflow-y-auto flex-1 bg-black/20 backdrop-blur-sm">
              {allHistory.length === 0 && savedGestures.length === 0 ? (
                <div className="text-center py-12">
                  <div className="bg-gradient-to-br from-orange-500/20 to-amber-500/20 rounded-full w-24 h-24 flex items-center justify-center mx-auto mb-6 border border-orange-500/30">
                    <History size={48} className="text-orange-400" />
                  </div>
                  <p className="text-gray-300 text-lg mb-2">No history available yet</p>
                  <p className="text-gray-400 text-sm">Start using live mode or gesture detection to see your history</p>
                </div>
              ) : (
                <div className="space-y-6">
                  {/* Transactions Section with Sort Options */}
                  <div>
                    <div className="flex items-center justify-between mb-4">
                      <h3 className="text-xl font-semibold text-white flex items-center">
                        <div className="bg-gradient-to-r from-blue-500/20 to-cyan-500/20 rounded-lg p-2 mr-3 border border-blue-500/30">
                          <History className="text-blue-400" size={20} />
                        </div>
                        <span className="bg-gradient-to-r from-blue-400 via-cyan-500 to-blue-600 bg-clip-text text-transparent">
                          All Transactions ({historyFilter === 'all' ? allHistory.length : allHistory.filter(item => item.type === historyFilter).length})
                        </span>
                      </h3>
                      {/* Sort/Filter Options */}
                      <div className="flex items-center gap-2">
                        <span className="text-gray-400 text-sm">Filter:</span>
                        <div 
                          className="relative"
                          onMouseEnter={() => setIsFilterDropdownOpen(true)}
                          onMouseLeave={() => setIsFilterDropdownOpen(false)}
                        >
                          <button className="bg-gradient-to-r from-orange-500/20 to-amber-500/20 border border-orange-500/30 rounded-lg px-4 py-2 text-orange-300 text-sm focus:outline-none focus:ring-2 focus:ring-orange-500/50 focus:border-orange-500/50 hover:from-orange-500/30 hover:to-amber-500/30 transition-all duration-300 flex items-center gap-2 backdrop-blur-sm">
                            {historyFilter === 'all' ? 'All Transactions' : 
                             historyFilter === 'lip-reading' ? 'Lip Reading Only' : 
                             'Gesture Text Only'}
                            <ChevronDown size={16} className={`transition-transform duration-200 ${isFilterDropdownOpen ? 'rotate-180' : ''}`} />
                          </button>
                          
                          {/* Dropdown Menu */}
                          <div className={`absolute top-full left-0 mt-1 w-48 bg-gradient-to-br from-gray-900/95 via-black/90 to-gray-800/95 backdrop-blur-xl border border-orange-500/30 rounded-lg shadow-xl z-50 overflow-hidden transition-all duration-300 ${
                            isFilterDropdownOpen ? 'opacity-100 translate-y-0 pointer-events-auto animate-dropdown-in' : 'opacity-0 -translate-y-2 pointer-events-none'
                          }`}>
                            <div className="py-1">
                              <button
                                onClick={() => setHistoryFilter('all')}
                                className={`w-full text-left px-4 py-3 text-sm transition-all duration-200 flex items-center gap-2 ${
                                  historyFilter === 'all' 
                                    ? 'bg-gradient-to-r from-orange-500/30 to-amber-500/30 text-orange-300 border-l-2 border-orange-500' 
                                    : 'text-gray-300 hover:bg-gradient-to-r hover:from-orange-500/20 hover:to-amber-500/20 hover:text-orange-200'
                                }`}
                              >
                                All Transactions
                                {historyFilter === 'all' && <CheckCircle2 size={16} className="ml-auto text-orange-400" />}
                              </button>
                              <button
                                onClick={() => setHistoryFilter('lip-reading')}
                                className={`w-full text-left px-4 py-3 text-sm transition-all duration-200 flex items-center gap-2 ${
                                  historyFilter === 'lip-reading' 
                                    ? 'bg-gradient-to-r from-orange-500/30 to-amber-500/30 text-orange-300 border-l-2 border-orange-500' 
                                    : 'text-gray-300 hover:bg-gradient-to-r hover:from-orange-500/20 hover:to-amber-500/20 hover:text-orange-200'
                                }`}
                              >
                                Lip Reading Only
                                {historyFilter === 'lip-reading' && <CheckCircle2 size={16} className="ml-auto text-orange-400" />}
                              </button>
                              <button
                                onClick={() => setHistoryFilter('gesture')}
                                className={`w-full text-left px-4 py-3 text-sm transition-all duration-200 flex items-center gap-2 ${
                                  historyFilter === 'gesture' 
                                    ? 'bg-gradient-to-r from-orange-500/30 to-amber-500/30 text-orange-300 border-l-2 border-orange-500' 
                                    : 'text-gray-300 hover:bg-gradient-to-r hover:from-orange-500/20 hover:to-amber-500/20 hover:text-orange-200'
                                }`}
                              >
                                Gesture Text Only
                                {historyFilter === 'gesture' && <CheckCircle2 size={16} className="ml-auto text-orange-400" />}
                              </button>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                    <div className="grid gap-4">
                      {(historyFilter === 'all' ? allHistory : allHistory.filter(item => item.type === historyFilter)).map((item) => (
                        <div key={item.id} className="rounded-xl p-6 border backdrop-blur-lg relative overflow-hidden transition-all duration-300 bg-gradient-to-br from-gray-900/50 via-black/30 to-gray-800/50 border-blue-500/20 hover:border-blue-500/40">
                          {/* Card decoration */}
                          <div className="absolute top-0 right-0 w-20 h-20 rounded-bl-xl bg-gradient-to-br from-blue-500/10 to-transparent"></div>
                          <div className="absolute bottom-0 left-0 w-16 h-16 rounded-tr-xl bg-gradient-to-tr from-cyan-500/10 to-transparent"></div>
                          <div className="relative z-10 flex justify-between items-start mb-2">
                            <div className="flex-1">
                              <div className="flex items-center mb-3">
                                <span className={`inline-flex items-center px-3 py-1 rounded-full text-xs font-semibold mr-2 border ${item.type === 'lip-reading' ? 'bg-orange-500/20 text-orange-300 border-orange-500/30' : 'bg-purple-500/20 text-purple-300 border-purple-500/30'}`}>
                                  {item.type === 'lip-reading' ? 'Lip Reading' : 'Gesture Text'}
                                </span>
                              </div>
                              <p className="text-gray-200 mb-3 leading-relaxed text-sm sm:text-base">{item.text}</p>
                              <div className="flex items-center justify-between text-sm">
                                <span className="text-gray-400">{item.timestamp}</span>
                                <span className={`font-medium ${item.type === 'lip-reading' ? 'text-orange-400' : 'text-purple-400'}`}>Confidence: {item.confidence}%</span>
                              </div>
                            </div>
                          </div>
                          <div className="relative z-10 flex items-center space-x-2 mt-4">
                            <button onClick={() => downloadSingleText(item.text, item.timestamp)} className="flex items-center space-x-1 px-3 py-2 bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 text-white text-sm rounded-lg transition-all duration-300 border border-blue-500/30">
                              <Download size={14} />
                              <span>Download</span>
                            </button>
                            <button onClick={() => deleteFromHistory(item.id)} className="flex items-center space-x-1 px-3 py-2 bg-gradient-to-r from-red-600 to-red-700 hover:from-red-700 hover:to-red-800 text-white text-sm rounded-lg transition-all duration-300 border border-red-500/30">
                              <Trash2 size={14} />
                              <span>Delete</span>
                            </button>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Modal Footer */}
            {allHistory.length > 0 && (
              <div className="relative z-10 border-t border-orange-500/20 p-4 bg-gradient-to-r from-black/60 via-gray-900/80 to-black/60 backdrop-blur-lg">
                <div className="flex items-center justify-between">
                  <span className="text-gray-300 text-sm">
                    Total: <span className="text-orange-400 font-medium">{allHistory.length}</span> transcription{allHistory.length !== 1 ? 's' : ''} | Saved: <span className="text-green-400 font-medium">{savedTexts.length}</span>
                  </span>
                  <div className="flex space-x-2">
                    <button
                      onClick={() => {
                        showToast('Download started', 'info');
                        const allTexts = allHistory
                          .map((item) => `[${item.timestamp}] (Confidence: ${item.confidence}%)\n${item.text}\n`)
                          .join('\n');
                        const blob = new Blob([allTexts], { type: 'text/plain' });
                        const url = URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = `complete-history-${new Date().toISOString().slice(0, 10)}.txt`;
                        document.body.appendChild(a);
                        a.click();
                        document.body.removeChild(a);
                        URL.revokeObjectURL(url);
                      }}
                      className="flex items-center space-x-1 px-4 py-2 bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 text-white text-sm rounded-lg transition-all duration-300 border border-blue-500/30"
                    >
                      <Download size={16} />
                      <span>Download All</span>
                    </button>
                    <button
                      onClick={() => {
                        showConfirmDialog('Are you sure you want to clear all history? This cannot be undone.', () => {
                          setAllHistory([]);
                          setSavedTexts([]);
                          setIsHistoryModalOpen(false);
                          showToast('History cleared', 'info');
                        });
                      }}
                      className="flex items-center space-x-1 px-4 py-2 bg-gradient-to-r from-red-600 to-red-700 hover:from-red-700 hover:to-red-800 text-white text-sm rounded-lg transition-all duration-300 border border-red-500/30"
                    >
                      <Trash2 size={16} />
                      <span>Clear All</span>
                    </button>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Custom Confirmation Dialog */}
      {isConfirmDialogOpen && (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-gray-900 rounded-xl border border-white/10 max-w-md w-full shadow-2xl">
            <div className="p-6">
              <h3 className="text-lg font-semibold text-white mb-4">Confirm Action</h3>
              <p className="text-gray-300 mb-6">{confirmMessage}</p>
              <div className="flex space-x-3 justify-end">
                <button onClick={handleCancel} className="px-4 py-2 bg-gray-600 hover:bg-gray-700 text-white rounded-lg transition-colors">
                  Cancel
                </button>
                <button onClick={handleConfirm} className="px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg transition-colors">
                  Confirm
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Demo Video Modal */}
      {isVideoModalOpen && (
        <div className="fixed inset-0 bg-black/90 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div 
            className="relative w-full max-w-3xl max-h-[80vh] bg-gradient-to-br from-slate-900 via-black to-gray-900 backdrop-blur-lg rounded-2xl border border-orange-500/20 shadow-2xl overflow-hidden flex flex-col"
            onMouseEnter={() => setIsControlsVisible(true)}
            onMouseLeave={() => setIsControlsVisible(false)}
          >
            {/* Professional background graphics */}
            <div className="absolute inset-0">
              {/* Gradient orbs for demo video theme */}
              <div className="absolute top-10 right-10 w-32 h-32 bg-gradient-to-r from-orange-500/20 to-amber-600/20 rounded-full blur-2xl animate-pulse"></div>
              <div className="absolute bottom-10 left-10 w-28 h-28 bg-gradient-to-r from-amber-500/15 to-yellow-500/15 rounded-full blur-2xl animate-pulse delay-1000"></div>
              
              {/* Geometric decorations */}
              <div className="absolute top-8 left-8 w-16 h-16 border border-orange-400/25 rounded-lg transform rotate-12"></div>
              <div className="absolute bottom-8 right-8 w-12 h-12 bg-gradient-to-br from-orange-500/10 to-transparent rounded-full"></div>
              
              {/* Dot pattern overlay */}
              <div className="absolute inset-0 opacity-[0.02]" style={{
                backgroundImage: `radial-gradient(circle at 2px 2px, orange 1px, transparent 0)`,
                backgroundSize: '20px 20px'
              }}></div>
            </div>

            {/* Modal Header */}
            <div className={`relative z-10 flex items-center justify-between p-4 border-b backdrop-blur-lg transition-all duration-500 ${
              isControlsVisible 
                ? 'border-orange-500/20 bg-gradient-to-r from-black/60 via-gray-900/80 to-black/60' 
                : 'border-orange-500/10 bg-gradient-to-r from-black/40 via-gray-900/60 to-black/40'
            }`}>
              <div className="flex items-center space-x-3">
                <div className="p-1.5 bg-gradient-to-r from-orange-500/20 to-amber-500/20 rounded-lg border border-orange-500/30">
                  <Play className="text-orange-400" size={20} />
                </div>
                <div>
                  <h2 className="text-xl font-bold text-white">
                    <span className="bg-gradient-to-r from-orange-400 via-amber-500 to-orange-600 bg-clip-text text-transparent">
                      Hand Gesture Demo
                    </span>
                  </h2>
                  <p className="text-orange-300 text-sm">Learn how to perform gestures correctly</p>
                </div>
              </div>
              <button 
                onClick={closeVideoModal}
                className="p-2 hover:bg-red-500/20 rounded-lg transition-all duration-300 group border border-transparent hover:border-red-500/30"
              >
                <X className="text-gray-400 group-hover:text-red-400" size={24} />
              </button>
            </div>

            {/* Video Content */}
            <div className="relative flex-1 flex flex-col">
              <video
                ref={demoVideoRef}
                className="w-full flex-1 object-contain bg-black"
                controls={false}
                autoPlay
                onPlay={() => setIsVideoPlaying(true)}
                onPause={() => setIsVideoPlaying(false)}
                onEnded={() => setIsVideoPlaying(false)}
                onLoadedData={() => {
                  if (demoVideoRef.current) {
                    demoVideoRef.current.playbackRate = videoPlaybackSpeed;
                  }
                }}
              >
                <source src="/demo-video.mp4" type="video/mp4" />
                <source src="/demo-video.webm" type="video/webm" />
                <div className="flex items-center justify-center h-64 text-white">
                  <div className="text-center">
                    <AlertTriangle size={48} className="mx-auto mb-4 text-yellow-400" />
                    <p className="text-lg mb-2">Demo video not found</p>
                    <p className="text-sm text-gray-400">
                      Please add your demo video as "demo-video.mp4" in the public folder
                    </p>
                  </div>
                </div>
              </video>

              {/* Enhanced Video Controls */}
              <div className={`absolute bottom-2 left-2 right-2 transition-all duration-500 ${
                isControlsVisible ? 'opacity-100 translate-y-0' : 'opacity-20 translate-y-2'
              }`}>
                <div className={`backdrop-blur-sm rounded-lg p-3 space-y-2 border transition-all duration-500 ${
                  isControlsVisible 
                    ? 'bg-black/80 border-orange-500/20' 
                    : 'bg-black/40 border-orange-500/10'
                }`}>
                  {/* Hover Indicator */}
                  {!isControlsVisible && (
                    <div className="absolute -top-2 left-1/2 transform -translate-x-1/2 -translate-y-full">
                      <div className="bg-orange-500/80 text-white text-xs px-2 py-1 rounded-md opacity-60 animate-pulse">
                        Hover for controls
                      </div>
                    </div>
                  )}
                  {/* Top Row - Title and Close */}
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-white text-sm font-medium">Hand Gesture Tutorial</p>
                      <p className="text-orange-300 text-xs">Follow along to learn proper gestures</p>
                    </div>
                    <button
                      onClick={closeVideoModal}
                      className="bg-gradient-to-r from-red-500/80 to-red-600/80 hover:from-red-600 hover:to-red-700 backdrop-blur-sm rounded-lg px-3 py-1.5 text-white text-sm font-medium transition-all duration-300 border border-red-500/30"
                    >
                      Close Demo
                    </button>
                  </div>
                  
                  {/* Bottom Row - Playback Controls */}
                  <div className="flex items-center justify-between">
                    {/* Left Side - Playback Controls */}
                    <div className="flex items-center space-x-3">
                      <button
                        onClick={handleVideoPlayPause}
                        className="bg-gradient-to-r from-orange-500/20 to-amber-500/20 hover:from-orange-500/40 hover:to-amber-500/40 rounded-lg p-2 text-orange-400 transition-all duration-300 border border-orange-500/30"
                        title={isVideoPlaying ? "Pause" : "Play"}
                      >
                        {isVideoPlaying ? <Pause size={20} /> : <Play size={20} />}
                      </button>
                      
                      <button
                        onClick={handleVideoReplay}
                        className="bg-gradient-to-r from-amber-500/20 to-yellow-500/20 hover:from-amber-500/40 hover:to-yellow-500/40 rounded-lg p-2 text-amber-400 transition-all duration-300 border border-amber-500/30"
                        title="Replay from beginning"
                      >
                        <RotateCw size={20} />
                      </button>
                      
                      <button
                        onClick={() => {
                          if (demoVideoRef.current) {
                            demoVideoRef.current.currentTime = Math.max(0, demoVideoRef.current.currentTime - 10);
                          }
                        }}
                        className="bg-gradient-to-r from-orange-600/20 to-red-500/20 hover:from-orange-600/40 hover:to-red-500/40 rounded-lg p-2 text-orange-300 transition-all duration-300 border border-orange-600/30"
                        title="Rewind 10 seconds"
                      >
                        <SkipBack size={20} />
                      </button>
                    </div>
                    
                    {/* Right Side - Speed Control */}
                    <div className="flex items-center space-x-2">
                      <span className="text-orange-300 text-sm">Speed:</span>
                      <div className="flex items-center space-x-1">
                        {[0.5, 0.75, 1, 1.25, 1.5, 2].map((speed) => (
                          <button
                            key={speed}
                            onClick={() => handlePlaybackSpeedChange(speed)}
                            className={`px-2 py-1 rounded text-xs font-medium transition-all duration-300 ${
                              videoPlaybackSpeed === speed
                                ? 'bg-gradient-to-r from-orange-500 to-amber-500 text-white border border-orange-400/50'
                                : 'bg-gray-600/50 text-gray-300 hover:bg-orange-500/20 hover:text-orange-400 border border-transparent hover:border-orange-500/30'
                            }`}
                            title={`Playback speed ${speed}x`}
                          >
                            {speed}x
                          </button>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Modal Footer with Instructions */}
            <div className={`relative z-10 p-4 backdrop-blur-lg border-t transition-all duration-500 ${
              isControlsVisible 
                ? 'bg-gradient-to-r from-black/60 via-gray-900/80 to-black/60 border-orange-500/20' 
                : 'bg-gradient-to-r from-black/40 via-gray-900/60 to-black/40 border-orange-500/10'
            }`}>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-sm mb-3">
                <div className="flex items-center space-x-2">
                  <div className="w-6 h-6 bg-gradient-to-r from-orange-500/20 to-amber-500/20 rounded-full flex items-center justify-center border border-orange-500/30">
                    <span className="text-orange-400 font-bold text-xs">1</span>
                  </div>
                  <div>
                    <p className="text-white font-medium text-sm">Position Your Hand</p>
                    <p className="text-orange-300 text-xs">Keep hand clearly visible in camera</p>
                  </div>
                </div>
                <div className="flex items-center space-x-2">
                  <div className="w-6 h-6 bg-gradient-to-r from-amber-500/20 to-yellow-500/20 rounded-full flex items-center justify-center border border-amber-500/30">
                    <span className="text-amber-400 font-bold text-xs">2</span>
                  </div>
                  <div>
                    <p className="text-white font-medium text-sm">Make Clear Gestures</p>
                    <p className="text-amber-300 text-xs">Follow the demo movements</p>
                  </div>
                </div>
                <div className="flex items-center space-x-2">
                  <div className="w-6 h-6 bg-gradient-to-r from-orange-600/20 to-red-500/20 rounded-full flex items-center justify-center border border-orange-600/30">
                    <span className="text-orange-300 font-bold text-xs">3</span>
                  </div>
                  <div>
                    <p className="text-white font-medium text-sm">Practice & Detect</p>
                    <p className="text-orange-200 text-xs">Start detection after watching</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default LipReadingApp;