import axios from 'axios';

const API_BASE_URL = 'https://sp-project-group-d-1.onrender.com';


const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const predictRisk = async (patientData) => {
  try {
    const response = await api.post('/predict', patientData);
    return response.data;
  } catch (error) {
    console.error('Error predicting risk:', error);
    throw error;
  }
};

export const checkHealth = async () => {
  try {
    const response = await api.get('/health');
    return response.data;
  } catch (error) {
    console.error('Error checking API health:', error);
    throw error;
  }
};