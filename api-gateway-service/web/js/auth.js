// Authentication module for AI Services Gateway
class AuthManager {
    constructor() {
        this.token = localStorage.getItem(CONFIG.STORAGE_KEYS.ACCESS_TOKEN);
        this.currentUser = JSON.parse(localStorage.getItem(CONFIG.STORAGE_KEYS.CURRENT_USER) || 'null');
    }

    // Check if user is authenticated
    isAuthenticated() {
        return !!this.token;
    }

    // Get current auth token
    getToken() {
        return this.token;
    }

    // Get current user
    getCurrentUser() {
        return this.currentUser;
    }

    // Set authentication data
    setAuth(token, user) {
        this.token = token;
        this.currentUser = user;
        localStorage.setItem(CONFIG.STORAGE_KEYS.ACCESS_TOKEN, token);
        localStorage.setItem(CONFIG.STORAGE_KEYS.CURRENT_USER, JSON.stringify(user));
    }

    // Clear authentication data
    clearAuth() {
        this.token = null;
        this.currentUser = null;
        localStorage.removeItem(CONFIG.STORAGE_KEYS.ACCESS_TOKEN);
        localStorage.removeItem(CONFIG.STORAGE_KEYS.CURRENT_USER);
        localStorage.removeItem(CONFIG.STORAGE_KEYS.CHAT_HISTORY);
    }

    // Login user
    async login(username, password) {
        try {
            const formData = new FormData();
            formData.append('username', username);
            formData.append('password', password);

            const response = await fetch(`${CONFIG.API_BASE_URL}${CONFIG.ENDPOINTS.LOGIN}`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.detail || `登录失败: ${response.status}`);
            }

            const data = await response.json();
            
            // Create user object
            const user = {
                username: username,
                loginTime: new Date().toISOString()
            };

            // Set authentication
            this.setAuth(data.access_token, user);

            return {
                success: true,
                token: data.access_token,
                user: user
            };

        } catch (error) {
            console.error('Login error:', error);
            return {
                success: false,
                error: error.message
            };
        }
    }

    // Logout user
    logout() {
        this.clearAuth();
        
        // Reset app state
        if (window.app) {
            window.app.reset();
        }
        
        // Redirect to login by hiding main app and showing login modal
        document.getElementById('app').style.display = 'none';
        document.getElementById('loginModal').style.display = 'flex';
        Utils.showToast('已退出登录', 'info');
    }

    // Get authorization headers for API calls
    getAuthHeaders() {
        if (!this.token) {
            return {};
        }
        return {
            'Authorization': `Bearer ${this.token}`
        };
    }

    // Check if token is expired (basic check)
    isTokenExpired() {
        if (!this.token) return true;

        try {
            // Basic JWT payload decode (without verification)
            const payload = JSON.parse(atob(this.token.split('.')[1]));
            const currentTime = Date.now() / 1000;
            
            return payload.exp && payload.exp < currentTime;
        } catch (error) {
            console.error('Error checking token expiration:', error);
            return true;
        }
    }

    // Auto-refresh token if needed (simplified version)
    async ensureValidToken() {
        if (!this.token) {
            throw new Error('No authentication token');
        }

        if (this.isTokenExpired()) {
            // In a real implementation, you might want to refresh the token
            // For now, we'll just logout
            this.logout();
            throw new Error('Token expired');
        }

        return this.token;
    }

    // Handle API response for auth errors
    handleApiResponse(response) {
        if (response.status === 401) {
            Utils.showToast('认证已过期，请重新登录', 'warning');
            this.logout();
            return false;
        }
        return true;
    }
}

// Login form handler
class LoginForm {
    constructor(authManager) {
        this.authManager = authManager;
        this.form = document.getElementById('loginForm');
        this.usernameInput = document.getElementById('username');
        this.passwordInput = document.getElementById('password');
        this.errorElement = document.getElementById('loginError');
        this.loginBtn = this.form.querySelector('.login-btn');

        this.initEventListeners();
        this.prefillCredentials();
    }

    initEventListeners() {
        this.form.addEventListener('submit', this.handleSubmit.bind(this));
        
        // Enter key handling
        this.passwordInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.handleSubmit(e);
            }
        });
    }

    prefillCredentials() {
        // Pre-fill with default credentials for easier testing
        this.usernameInput.value = CONFIG.DEFAULT_CREDENTIALS.username;
        this.passwordInput.value = CONFIG.DEFAULT_CREDENTIALS.password;
    }

    async handleSubmit(e) {
        e.preventDefault();

        const username = this.usernameInput.value.trim();
        const password = this.passwordInput.value;

        if (!username || !password) {
            this.showError('请输入用户名和密码');
            return;
        }

        this.setLoading(true);
        this.hideError();

        try {
            const result = await this.authManager.login(username, password);

            if (result.success) {
                this.hideModal();
                this.showMainApp();
                Utils.showToast('登录成功！', 'success');
                
                // Initialize the app after successful login
                if (window.app && !window.app.isInitialized) {
                    await window.app.init();
                }
            } else {
                this.showError(result.error);
            }
        } catch (error) {
            console.error('Login error:', error);
            this.showError('登录时发生错误，请稍后重试');
        } finally {
            this.setLoading(false);
        }
    }

    showError(message) {
        this.errorElement.textContent = message;
        this.errorElement.style.display = 'block';
    }

    hideError() {
        this.errorElement.style.display = 'none';
    }

    setLoading(loading) {
        this.loginBtn.disabled = loading;
        if (loading) {
            this.loginBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> 登录中...';
        } else {
            this.loginBtn.innerHTML = '<i class="fas fa-sign-in-alt"></i> 登录';
        }
    }

    hideModal() {
        document.getElementById('loginModal').style.display = 'none';
    }

    showMainApp() {
        document.getElementById('app').style.display = 'flex';
    }
}

// Initialize authentication
const authManager = new AuthManager();

// Export for global use
window.authManager = authManager;

// Auto-initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    // Initialize login form
    window.loginForm = new LoginForm(authManager);

    // Check if already authenticated
    if (authManager.isAuthenticated() && !authManager.isTokenExpired()) {
        // Hide login modal and show main app
        document.getElementById('loginModal').style.display = 'none';
        document.getElementById('app').style.display = 'flex';
        
        // Initialize app if available and not already initialized
        if (window.app && window.app.init && !window.app.isInitialized) {
            window.app.init();
        }
    } else {
        // Show login modal
        document.getElementById('loginModal').style.display = 'flex';
        document.getElementById('app').style.display = 'none';
        
        // Clear any invalid auth data
        authManager.clearAuth();
    }

    // Setup logout button
    const logoutBtn = document.getElementById('logoutBtn');
    if (logoutBtn) {
        logoutBtn.addEventListener('click', () => {
            authManager.logout();
        });
    }
});