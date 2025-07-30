// Main application entry point for AI Services Gateway
class App {
    constructor() {
        this.authManager = window.authManager;
        this.apiClient = null;
        this.chatManager = null;
        this.isInitialized = false;
    }

    // Initialize the application
    async init() {
        if (this.isInitialized) return;

        try {
            console.log('Initializing AI Services Gateway...');

            // Ensure user is authenticated
            if (!this.authManager.isAuthenticated()) {
                console.log('User not authenticated, showing login modal');
                this.showLogin();
                return;
            }

            // Initialize API client
            this.apiClient = new APIClient(this.authManager);

            // Initialize users dropdown first
            await this.initUsersDropdown();

            // Initialize chat manager after users are loaded
            this.chatManager = new ChatManager(this.apiClient);
            await this.chatManager.init();

            // Set up periodic token refresh
            this.setupTokenRefresh();

            // Mark as initialized
            this.isInitialized = true;

            console.log('AI Services Gateway initialized successfully');
            Utils.showToast('应用初始化成功', 'success');

        } catch (error) {
            console.error('Failed to initialize application:', error);
            Utils.showToast('应用初始化失败', 'error');
            
            // If initialization fails due to auth, show login
            if (error.message.includes('Authentication') || error.message.includes('Token')) {
                this.authManager.logout();
            }
        }
    }

    // Initialize users dropdown with available users
    async initUsersDropdown() {
        try {
            // Get all users from the system
            const usersResponse = await this.apiClient.getAllUsers();
            const users = usersResponse.users || [];

            const userSelect = document.getElementById('userUID');
            if (!userSelect) return;

            // Clear existing options
            userSelect.innerHTML = '';

            // Add default users if no users returned
            if (users.length === 0) {
                const defaultUsers = [
                    { user_id: 'user_001', name: '用户001' },
                    { user_id: 'user_002', name: '用户002' },
                    { user_id: 'user_003', name: '用户003' }
                ];

                defaultUsers.forEach(user => {
                    const option = document.createElement('option');
                    option.value = user.user_id;
                    option.textContent = Utils.truncateUserId(user.name);
                    userSelect.appendChild(option);
                });

                // Try to add default users to the system
                for (const user of defaultUsers) {
                    try {
                        await this.apiClient.addUser(user.user_id);
                    } catch (error) {
                        // Ignore errors (user might already exist)
                        console.log(`User ${user.user_id} might already exist`);
                    }
                }
            } else {
                // Add actual users from system
                // Handle both string arrays and object arrays
                users.forEach(user => {
                    const option = document.createElement('option');
                    if (typeof user === 'string') {
                        // User is a string ID
                        option.value = user;
                        option.textContent = Utils.truncateUserId(user);
                    } else {
                        // User is an object
                        option.value = user.user_id || user.id;
                        option.textContent = Utils.truncateUserId(user.name || user.user_id || user.id);
                    }
                    userSelect.appendChild(option);
                });
            }

            // Set first user as selected if none is selected
            if (userSelect.children.length > 0 && !userSelect.value) {
                userSelect.selectedIndex = 0;
            }
            
            console.log('Users dropdown initialized with', userSelect.children.length, 'users');

        } catch (error) {
            console.error('Failed to initialize users dropdown:', error);
            // Fallback to default users
            this.createDefaultUsersDropdown();
        }
    }

    // Create default users dropdown as fallback
    createDefaultUsersDropdown() {
        const userSelect = document.getElementById('userUID');
        if (!userSelect) return;

        userSelect.innerHTML = '';
        
        const defaultUsers = [
            { id: 'user_001', name: '用户001' },
            { id: 'user_002', name: '用户002' },
            { id: 'user_003', name: '用户003' }
        ];

        defaultUsers.forEach(user => {
            const option = document.createElement('option');
            option.value = user.id;
            option.textContent = Utils.truncateUserId(user.name);
            userSelect.appendChild(option);
        });
        
        // Set first user as selected
        if (userSelect.children.length > 0) {
            userSelect.selectedIndex = 0;
        }
    }

    // Setup periodic token refresh
    setupTokenRefresh() {
        // Check token validity every 5 minutes
        setInterval(() => {
            if (this.authManager.isTokenExpired()) {
                console.log('Token expired, logging out');
                Utils.showToast('会话已过期，请重新登录', 'warning');
                this.authManager.logout();
            }
        }, 5 * 60 * 1000);
    }

    // Show login modal
    showLogin() {
        document.getElementById('loginModal').style.display = 'flex';
        document.getElementById('app').style.display = 'none';
        this.isInitialized = false;
    }

    // Show main application
    showApp() {
        document.getElementById('loginModal').style.display = 'none';
        document.getElementById('app').style.display = 'flex';
    }

    // Handle application errors
    handleError(error, context = 'Application') {
        console.error(`${context} error:`, error);
        
        if (error.message.includes('Authentication') || 
            error.message.includes('Token') || 
            error.message.includes('401')) {
            Utils.showToast('认证失败，请重新登录', 'error');
            this.authManager.logout();
        } else {
            Utils.showToast(`${context}出错: ${error.message}`, 'error');
        }
    }

    // Get current application state
    getState() {
        return {
            isInitialized: this.isInitialized,
            isAuthenticated: this.authManager.isAuthenticated(),
            currentUser: this.authManager.getCurrentUser(),
            currentUserId: this.chatManager ? this.chatManager.currentUserId : null,
            currentChatId: this.chatManager ? this.chatManager.currentChatId : null
        };
    }

    // Reset application state
    reset() {
        this.isInitialized = false;
        this.apiClient = null;
        this.chatManager = null;
        // Clear global chat manager instance
        if (window.chatManagerInstance) {
            window.chatManagerInstance = null;
        }
    }
}

// Global error handler
window.addEventListener('error', (event) => {
    console.error('Global error:', event.error);
    if (window.app) {
        window.app.handleError(event.error, 'Global');
    }
});

// Handle unhandled promise rejections
window.addEventListener('unhandledrejection', (event) => {
    console.error('Unhandled promise rejection:', event.reason);
    if (window.app) {
        window.app.handleError(event.reason, 'Promise');
    }
});

// Initialize application when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM loaded, initializing application...');
    
    // Create global app instance
    window.app = new App();
    
    // Initialize if user is already authenticated
    if (window.authManager && window.authManager.isAuthenticated()) {
        window.app.init();
    }
    
    console.log('Application setup complete');
});

// Handle page visibility changes
document.addEventListener('visibilitychange', () => {
    if (!document.hidden && window.app && window.authManager) {
        // Page became visible, check if token is still valid
        if (window.authManager.isAuthenticated() && window.authManager.isTokenExpired()) {
            Utils.showToast('会话已过期，请重新登录', 'warning');
            window.authManager.logout();
        }
    }
});

// Export for global access
window.App = App;