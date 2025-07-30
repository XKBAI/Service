// API client for AI Services Gateway
class APIClient {
    constructor(authManager) {
        this.authManager = authManager;
        this.baseURL = CONFIG.API_BASE_URL;
    }

    // Make authenticated request
    async makeRequest(endpoint, options = {}) {
        try {
            // Ensure we have a valid token
            await this.authManager.ensureValidToken();

            const url = `${this.baseURL}${endpoint}`;
            const headers = {
                ...this.authManager.getAuthHeaders(),
                ...options.headers
            };

            const response = await fetch(url, {
                ...options,
                headers
            });

            // Handle auth errors
            if (!this.authManager.handleApiResponse(response)) {
                throw new Error('Authentication failed');
            }

            return response;
        } catch (error) {
            console.error('API request error:', error);
            throw error;
        }
    }

    // User Management APIs
    async getAllUsers(retryCount = 0, maxRetries = 3) {
        try {
            const response = await this.makeRequest(CONFIG.ENDPOINTS.USER.GET_ALL_USERS);
            if (!response.ok) throw new Error(`Failed to get users: ${response.status}`);
            
            try {
                return await response.json();
            } catch (jsonError) {
                console.warn('JSON parsing failed for getAllUsers:', jsonError);
                
                // If Content-Length error or similar network issue, retry after delay
                if ((jsonError.message.includes('Content-Length') || 
                     jsonError.message.includes('Unexpected end') ||
                     jsonError.message.includes('Parse error') ||
                     jsonError.message.includes('Failed to fetch')) && 
                    retryCount < maxRetries) {
                    
                    console.log(`Retrying getAllUsers request (${retryCount + 1}/${maxRetries + 1}) after error`);
                    await new Promise(resolve => setTimeout(resolve, 1000 * (retryCount + 1)));
                    return this.getAllUsers(retryCount + 1, maxRetries);
                }
                
                throw jsonError;
            }
        } catch (error) {
            console.error('Get all users error:', error);
            
            // Retry for network errors
            if ((error.message.includes('Content-Length') || 
                 error.message.includes('fetch') ||
                 error.message.includes('Failed to fetch') ||
                 error.message.includes('ERR_CONTENT_LENGTH_MISMATCH')) && 
                retryCount < maxRetries) {
                
                console.log(`Retrying getAllUsers due to network error (${retryCount + 1}/${maxRetries + 1})`);
                await new Promise(resolve => setTimeout(resolve, 2000 * (retryCount + 1)));
                return this.getAllUsers(retryCount + 1, maxRetries);
            }
            
            throw error;
        }
    }

    async addUser(userId) {
        try {
            const response = await this.makeRequest(CONFIG.ENDPOINTS.USER.ADD_USER, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ user_id: userId })
            });
            if (!response.ok) throw new Error(`Failed to add user: ${response.status}`);
            return await response.json();
        } catch (error) {
            console.error('Add user error:', error);
            throw error;
        }
    }

    async getUserChats(userId, retryCount = 0, maxRetries = 3) {
        try {
            const response = await this.makeRequest(`${CONFIG.ENDPOINTS.USER.GET_USER_CHAT}?user_id=${encodeURIComponent(userId)}`);
            if (!response.ok) throw new Error(`Failed to get user chats: ${response.status}`);
            
            try {
                return await response.json();
            } catch (jsonError) {
                console.warn('JSON parsing failed for getUserChats:', jsonError);
                
                // If Content-Length error or similar network issue, retry after delay
                if ((jsonError.message.includes('Content-Length') || 
                     jsonError.message.includes('Unexpected end') ||
                     jsonError.message.includes('Parse error') ||
                     jsonError.message.includes('Failed to fetch')) && 
                    retryCount < maxRetries) {
                    
                    console.log(`Retrying getUserChats request (${retryCount + 1}/${maxRetries + 1}) after Content-Length error`);
                    await new Promise(resolve => setTimeout(resolve, 1000 * (retryCount + 1))); // Exponential backoff
                    return this.getUserChats(userId, retryCount + 1, maxRetries);
                }
                
                throw jsonError;
            }
        } catch (error) {
            console.error('Get user chats error:', error);
            
            // Retry for network errors
            if ((error.message.includes('Content-Length') || 
                 error.message.includes('fetch') ||
                 error.message.includes('Failed to fetch') ||
                 error.message.includes('ERR_CONTENT_LENGTH_MISMATCH')) && 
                retryCount < maxRetries) {
                
                console.log(`Retrying getUserChats due to network error (${retryCount + 1}/${maxRetries + 1})`);
                await new Promise(resolve => setTimeout(resolve, 2000 * (retryCount + 1)));
                return this.getUserChats(userId, retryCount + 1, maxRetries);
            }
            
            throw error;
        }
    }

    async addChat(userId, chatContent) {
        try {
            const response = await this.makeRequest(`${CONFIG.ENDPOINTS.USER.ADD_CHAT}?user_id=${encodeURIComponent(userId)}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(chatContent)
            });
            if (!response.ok) throw new Error(`Failed to add chat: ${response.status}`);
            return await response.json();
        } catch (error) {
            console.error('Add chat error:', error);
            throw error;
        }
    }

    async updateChat(chatSession) {
        try {
            const response = await this.makeRequest(CONFIG.ENDPOINTS.USER.UPDATE_CHAT, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(chatSession)
            });
            if (!response.ok) throw new Error(`Failed to update chat: ${response.status}`);
            return await response.json();
        } catch (error) {
            console.error('Update chat error:', error);
            throw error;
        }
    }

    async getChatBySessionId(sessionId) {
        try {
            const response = await this.makeRequest(`${CONFIG.ENDPOINTS.USER.GET_CHAT_BY_SESSION_ID}?chat_session_id=${encodeURIComponent(sessionId)}`);
            if (!response.ok) throw new Error(`Failed to get chat: ${response.status}`);
            return await response.json();
        } catch (error) {
            console.error('Get chat by session ID error:', error);
            throw error;
        }
    }

    async deleteChat(sessionId) {
        try {
            const response = await this.makeRequest(CONFIG.ENDPOINTS.USER.DELETE_CHAT, {
                method: 'DELETE',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ chat_session_id: sessionId })
            });
            if (!response.ok) throw new Error(`Failed to delete chat: ${response.status}`);
            return await response.json();
        } catch (error) {
            console.error('Delete chat error:', error);
            throw error;
        }
    }

    async updateChatTitle(sessionId, title) {
        try {
            const response = await this.makeRequest(CONFIG.ENDPOINTS.USER.EDIT_CHAT_TITLE, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    chat_session_id: sessionId,
                    chat_title: title,
                    chat_time: new Date().toISOString()
                })
            });
            if (!response.ok) throw new Error(`Failed to update chat title: ${response.status}`);
            return await response.json();
        } catch (error) {
            console.error('Update chat title error:', error);
            throw error;
        }
    }

    // LLM APIs
    async chatCompletion(prompt, messages = [], signal = null, chatType = 'base') {
        try {
            const payload = {
                prompt: prompt,
                messages: messages,
                chat_type: chatType
            };

            console.log('Sending LLM request with payload:', payload);

            const requestOptions = {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'text/event-stream'
                },
                body: JSON.stringify(payload)
            };

            // Add abort signal if provided
            if (signal) {
                requestOptions.signal = signal;
            }

            const response = await this.makeRequest(CONFIG.ENDPOINTS.LLM.CHAT_COMPLETIONS, requestOptions);

            if (!response.ok) throw new Error(`Failed to get chat completion: ${response.status}`);
            
            // Return the response for streaming
            return response;
        } catch (error) {
            console.error('Chat completion error:', error);
            throw error;
        }
    }

    async getChatTitle(messages) {
        try {
            const payload = { messages: messages };

            const response = await this.makeRequest(CONFIG.ENDPOINTS.LLM.GET_CHAT_TITLE, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payload)
            });

            if (!response.ok) throw new Error(`Failed to get chat title: ${response.status}`);
            return await response.json();
        } catch (error) {
            console.error('Get chat title error:', error);
            throw error;
        }
    }

    // OCR APIs
    async processOCR(file) {
        try {
            console.log('Sending OCR request for file:', file.name, 'type:', file.type, 'size:', file.size);
            const formData = new FormData();
            formData.append('file', file);

            const response = await this.makeRequest(CONFIG.ENDPOINTS.OCR.PROCESS, {
                method: 'POST',
                body: formData
            });

            console.log('OCR response status:', response.status);
            if (!response.ok) {
                const errorText = await response.text();
                console.error('OCR request failed:', response.status, errorText);
                throw new Error(`Failed to process OCR: ${response.status} - ${errorText}`);
            }
            
            const result = await response.json();
            console.log('OCR response body:', result);
            return result;
        } catch (error) {
            console.error('OCR process error:', error);
            throw error;
        }
    }

    async getOCRResults(jobId, retryCount = 0, maxRetries = 3) {
        try {
            // Add retry logic for Content-Length errors
            const response = await this.makeRequest(`${CONFIG.ENDPOINTS.OCR.GET_RESULTS}${jobId}`);
            if (!response.ok) throw new Error(`Failed to get OCR results: ${response.status}`);
            
            try {
                const result = await response.json();
                return result;
            } catch (jsonError) {
                console.warn('JSON parsing failed for OCR results:', jsonError);
                
                // If Content-Length error or similar network issue, retry after delay
                if ((jsonError.message.includes('Content-Length') || 
                     jsonError.message.includes('Unexpected end') ||
                     jsonError.message.includes('Parse error')) && 
                    retryCount < maxRetries) {
                    
                    console.log(`Retrying OCR results request (${retryCount + 1}/${maxRetries + 1}) after Content-Length error`);
                    await new Promise(resolve => setTimeout(resolve, 1000 * (retryCount + 1))); // Exponential backoff
                    return this.getOCRResults(jobId, retryCount + 1, maxRetries);
                }
                
                // Don't make another request if we already have network issues
                console.error('OCR parsing failed after retries, skipping additional requests');
                throw new Error(`OCR响应解析失败，服务可能暂时不可用`);
            }
        } catch (error) {
            console.error('Get OCR results error:', error);
            
            // Retry for network errors
            if ((error.message.includes('Content-Length') || 
                 error.message.includes('fetch') ||
                 error.message.includes('ERR_CONTENT_LENGTH_MISMATCH')) && 
                retryCount < maxRetries) {
                
                console.log(`Retrying OCR request due to network error (${retryCount + 1}/${maxRetries + 1})`);
                await new Promise(resolve => setTimeout(resolve, 2000 * (retryCount + 1)));
                return this.getOCRResults(jobId, retryCount + 1, maxRetries);
            }
            
            throw error;
        }
    }

    // TTS APIs
    async synthesizeText(text, language = CONFIG.TTS.DEFAULT_LANGUAGE, speakerId = CONFIG.TTS.DEFAULT_SPEAKER, speed = CONFIG.TTS.DEFAULT_SPEED) {
        try {
            const payload = {
                text: text,
                language: language,
                speaker_id: speakerId,
                speed: speed
            };

            const response = await this.makeRequest(CONFIG.ENDPOINTS.TTS.SYNTHESIZE, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payload)
            });

            if (!response.ok) throw new Error(`Failed to synthesize text: ${response.status}`);
            
            // Return blob for audio playback
            return await response.blob();
        } catch (error) {
            console.error('TTS synthesis error:', error);
            throw error;
        }
    }

    // STT APIs
    async transcribeAudio(audioBlob, language = CONFIG.STT.DEFAULT_LANGUAGE) {
        try {
            console.log('Sending STT request for audio blob, size:', audioBlob.size);
            const formData = new FormData();
            formData.append('file', audioBlob, 'recording.wav');
            formData.append('language', language);

            const response = await this.makeRequest(CONFIG.ENDPOINTS.STT.TRANSCRIBE, {
                method: 'POST',
                body: formData
            });

            console.log('STT response status:', response.status);
            if (!response.ok) {
                const errorText = await response.text();
                console.error('STT request failed:', response.status, errorText);
                throw new Error(`Failed to transcribe audio: ${response.status} - ${errorText}`);
            }
            
            const result = await response.json();
            console.log('STT response body:', result);
            return result;
        } catch (error) {
            console.error('STT transcribe error:', error);
            throw error;
        }
    }

    // File processing helper
    async processFileForChat(file) {
        try {
            const validation = Utils.validateFile(file);
            if (!validation.valid) {
                throw new Error(validation.error);
            }

            // Check if it's an image or PDF
            if (file.type.startsWith('image/') || file.type === 'application/pdf') {
                // Process with OCR
                console.log('Starting OCR processing for file:', file.name);
                const ocrJob = await this.processOCR(file);
                console.log('OCR job response:', ocrJob);
                
                if (ocrJob.status === 'completed') {
                    // OCR is already completed, return results directly
                    console.log('OCR completed immediately');
                    return {
                        type: 'ocr',
                        content: ocrJob.text || '无法识别文件内容',
                        filename: file.name,
                        fileSize: Utils.formatFileSize(file.size)
                    };
                } else if (ocrJob.job_id || ocrJob.id) {
                    // OCR job is processing, poll for results
                    const jobId = ocrJob.job_id || ocrJob.id;
                    console.log('OCR job submitted, starting polling for job_id:', jobId);
                    return await this.pollOCRResults(jobId, file.name, file.size);
                } else {
                    console.error('Invalid OCR response format:', ocrJob);
                    throw new Error(`OCR处理失败: 无效的响应格式`);
                }
            } else {
                throw new Error('不支持的文件类型');
            }
        } catch (error) {
            console.error('File processing error:', error);
            
            // Provide fallback option for OCR failures
            if (error.message.includes('OCR') || error.message.includes('Content-Length')) {
                return {
                    type: 'file_error',
                    content: `[文件: ${file.name} (${Utils.formatFileSize(file.size)}) - OCR处理失败，但文件已上传。您可以描述文件内容或重新上传。]`,
                    filename: file.name,
                    fileSize: Utils.formatFileSize(file.size),
                    error: error.message
                };
            }
            
            throw error;
        }
    }

    // Poll OCR results until completion
    async pollOCRResults(jobId, filename, fileSize, maxAttempts = 60) {
        console.log('Starting OCR polling for job:', jobId);
        let consecutiveErrors = 0;
        
        for (let attempt = 0; attempt < maxAttempts; attempt++) {
            try {
                // Exponential backoff: start with 2s, increase on consecutive errors
                const baseDelay = 2000;
                const errorMultiplier = Math.min(consecutiveErrors, 3); // Cap at 3 for max 8s delay
                const delay = baseDelay + (errorMultiplier * 2000);
                
                await new Promise(resolve => setTimeout(resolve, delay));
                
                console.log(`OCR polling attempt ${attempt + 1}/${maxAttempts} for job:`, jobId);
                const results = await this.getOCRResults(jobId);
                console.log('OCR poll result:', results);
                
                // Reset error counter on successful request
                consecutiveErrors = 0;
                
                if (results.status === 'completed') {
                    console.log('OCR completed successfully');
                    return {
                        type: 'ocr',
                        content: results.text || '无法识别文件内容',
                        filename: filename,
                        fileSize: Utils.formatFileSize(fileSize)
                    };
                } else if (results.status === 'failed') {
                    throw new Error('OCR处理失败');
                }
                
                console.log(`OCR still processing, status: ${results.status}`);
                // Continue polling if still processing
            } catch (error) {
                consecutiveErrors++;
                console.error(`OCR polling error on attempt ${attempt + 1} (consecutive errors: ${consecutiveErrors}):`, error);
                
                // If we have too many consecutive errors, fail faster
                if (consecutiveErrors >= 5) {
                    console.error('Too many consecutive OCR polling errors, giving up');
                    throw new Error('OCR服务暂时不可用，请稍后重试');
                }
                
                if (attempt === maxAttempts - 1) {
                    throw new Error('OCR处理超时');
                }
                
                // Add extra delay for network errors
                if (error.message.includes('Failed to fetch') || error.message.includes('Content-Length')) {
                    console.log('Network error detected, adding extra delay...');
                    await new Promise(resolve => setTimeout(resolve, 3000));
                }
            }
        }
        
        throw new Error('OCR处理超时');
    }
}

// Export for global use
window.APIClient = APIClient;