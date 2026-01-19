// frontend/src/services/api.js
// API client con detección automática de entorno

// Configuración de URL del backend
// En producción usa la URL de Render, en desarrollo usa localhost
const API_BASE_URL = import.meta.env.VITE_API_URL || 
                     (import.meta.env.PROD 
                       ? 'https://sp-backend-qkjs.onrender.com' 
                       : 'http://localhost:8000');

console.log('🔗 API URL:', API_BASE_URL); // Para debugging

/**
 * Predice el riesgo de enfermedad coronaria
 * @param {Object} patientData - Datos del paciente
 * @returns {Promise<Object>} - Predicción con probabilidad y nivel de riesgo
 */
export const predictRisk = async (patientData) => {
  try {
    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(patientData),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error('❌ API Error:', error);
    throw error;
  }
};

/**
 * Verifica el estado del backend
 * @returns {Promise<boolean>} - true si el backend está disponible
 */
export const checkHealth = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/health`);
    const data = await response.json();
    console.log('✅ Backend health:', data);
    return response.ok && data.model_loaded;
  } catch (error) {
    console.error('❌ Health check failed:', error);
    return false;
  }
};