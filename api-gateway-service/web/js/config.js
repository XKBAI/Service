// Configuration file for the AI Services Gateway frontend
const CONFIG = {
    // API Base URL - adjust this based on your gateway deployment
    API_BASE_URL: 'http://localhost:60443',
    
    // Default credentials (should be changed in production)
    DEFAULT_CREDENTIALS: {
        username: 'xkbAI',
        password: 'XuekuibangAI@2025'
    },
    
    // API Endpoints
    ENDPOINTS: {
        LOGIN: '/token',
        USER: {
            GET_ALL_USERS: '/user/get_all_users/',
            ADD_USER: '/user/add_user/',
            DELETE_USER: '/user/del_user/',
            GET_USER_CHAT: '/user/get_user_chat/',
            ADD_CHAT: '/user/add_chat/',
            DELETE_CHAT: '/user/del_chat/',
            EDIT_CHAT: '/user/edit_chat/',
            UPDATE_CHAT: '/user/update_chat/',
            GET_CHAT_BY_SESSION_ID: '/user/get_chat_by_session_id/',
            UPDATE_CHAT_TITLE: '/user/update_chat_title/',
            EDIT_CHAT_TITLE: '/user/edit_chat_title/'
        },
        LLM: {
            CHAT_COMPLETIONS: '/llm/chat/completions',
            GET_CHAT_TITLE: '/llm/get_chat_title'
        },
        OCR: {
            PROCESS: '/olmocr/process',
            GET_RESULTS: '/olmocr/results/'
        },
        TTS: {
            SYNTHESIZE: '/tts/synthesize'
        },
        STT: {
            TRANSCRIBE: '/stt/transcribe'
        }
    },
    
    // Chat Types for LLM
    CHAT_TYPES: {
        DEFAULT: 'base',
        CREATIVE: 'creative',
        PRECISE: 'precise'
    },
    
    // File Upload Settings
    FILE_UPLOAD: {
        MAX_SIZE: 10 * 1024 * 1024, // 10MB
        ALLOWED_TYPES: {
            'application/pdf': 'PDF',
            'image/png': 'PNG',
            'image/jpeg': 'JPEG',
            'image/jpg': 'JPG',
            'image/gif': 'GIF',
            'image/bmp': 'BMP',
            'image/webp': 'WebP'
        }
    },
    
    // UI Settings
    UI: {
        MESSAGE_TYPING_SPEED: 30, // ms per character
        AUTO_SCROLL_OFFSET: 100,
        TOAST_DURATION: 3000,
        POLLING_INTERVAL: 5000 // for checking OCR job status
    },
    
    // TTS Settings
    TTS: {
        DEFAULT_LANGUAGE: 'ZH',
        DEFAULT_SPEAKER: 'ZH',
        DEFAULT_SPEED: 1.0
    },
    
    // STT Settings
    STT: {
        DEFAULT_LANGUAGE: 'zh',
        MAX_RECORDING_TIME: 60000, // 60 seconds
        AUDIO_CONSTRAINTS: {
            audio: {
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true,
                sampleRate: 16000
            }
        }
    },
    
    // Local Storage Keys
    STORAGE_KEYS: {
        ACCESS_TOKEN: 'ai_gateway_token',
        CURRENT_USER: 'ai_gateway_current_user',
        CHAT_HISTORY: 'ai_gateway_chat_history',
        USER_PREFERENCES: 'ai_gateway_preferences'
    }
};

// Utility functions
const Utils = {
    // Format timestamp
    formatTime: (timestamp) => {
        const date = new Date(timestamp);
        const now = new Date();
        const diffInHours = (now - date) / (1000 * 60 * 60);
        
        if (diffInHours < 1) {
            return '刚刚';
        } else if (diffInHours < 24) {
            return `${Math.floor(diffInHours)}小时前`;
        } else if (diffInHours < 48) {
            return '昨天';
        } else {
            return date.toLocaleDateString('zh-CN');
        }
    },
    
    // Generate unique ID
    generateId: () => {
        return Date.now().toString(36) + Math.random().toString(36).substr(2);
    },
    
    // Format file size
    formatFileSize: (bytes) => {
        if (bytes === 0) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    },
    
    // Truncate text
    truncateText: (text, maxLength = 50) => {
        if (text.length <= maxLength) return text;
        return text.substr(0, maxLength) + '...';
    },
    
    // Truncate user ID to consistent display length
    truncateUserId: (userId) => {
        if (!userId) return '';
        
        // Check if text contains Chinese characters
        const hasChinese = /[\u4e00-\u9fa5]/.test(userId);
        
        if (hasChinese) {
            // For Chinese: max 3 characters + ellipsis
            if (userId.length <= 3) return userId;
            return userId.substr(0, 3) + '...';
        } else {
            // For English/Numbers: max 6 characters + ellipsis  
            if (userId.length <= 6) return userId;
            return userId.substr(0, 6) + '...';
        }
    },
    
    // Show toast notification
    showToast: (message, type = 'info') => {
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.textContent = message;
        
        // Add toast styles if not already present
        if (!document.getElementById('toast-styles')) {
            const style = document.createElement('style');
            style.id = 'toast-styles';
            style.textContent = `
                .toast {
                    position: fixed;
                    top: 20px;
                    right: 20px;
                    padding: 12px 20px;
                    border-radius: 8px;
                    color: white;
                    font-weight: 500;
                    z-index: 3000;
                    opacity: 0;
                    transform: translateX(100%);
                    transition: all 0.3s ease;
                }
                .toast.show {
                    opacity: 1;
                    transform: translateX(0);
                }
                .toast-info { background: #3498db; }
                .toast-success { background: #2ecc71; }
                .toast-warning { background: #f39c12; }
                .toast-error { background: #e74c3c; }
            `;
            document.head.appendChild(style);
        }
        
        document.body.appendChild(toast);
        
        // Show toast
        setTimeout(() => toast.classList.add('show'), 100);
        
        // Hide and remove toast
        setTimeout(() => {
            toast.classList.remove('show');
            setTimeout(() => document.body.removeChild(toast), 300);
        }, CONFIG.UI.TOAST_DURATION);
    },
    
    // Validate file
    validateFile: (file) => {
        if (file.size > CONFIG.FILE_UPLOAD.MAX_SIZE) {
            return {
                valid: false,
                error: `文件大小不能超过 ${Utils.formatFileSize(CONFIG.FILE_UPLOAD.MAX_SIZE)}`
            };
        }
        
        if (!CONFIG.FILE_UPLOAD.ALLOWED_TYPES[file.type]) {
            return {
                valid: false,
                error: '不支持的文件类型。请上传 PDF 或图片文件。'
            };
        }
        
        return { valid: true };
    }
};

// Export for use in other modules
window.CONFIG = CONFIG;
window.Utils = Utils;