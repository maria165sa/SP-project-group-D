import { useState } from 'react';
import PatientForm from './components/PatientForm';
import ResultDisplay from './components/ResultDisplay';
import { predictRisk } from './services/api';

function App() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (patientData) => {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const prediction = await predictRisk(patientData);
      setResult(prediction);
    } catch (err) {
      setError('Failed to get prediction. Make sure the API is running on http://localhost:8000');
      console.error('Error:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 py-8 px-4">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            Coronary Heart Disease Risk Predictor
          </h1>
          <p className="text-gray-600">
            10-Year Risk Assessment Based on Clinical Data
          </p>
        </div>

        {/* Error Message */}
        {error && (
          <div className="mb-6 bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded-lg">
            <p className="font-medium">Error</p>
            <p className="text-sm">{error}</p>
          </div>
        )}

        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Form Column */}
          <div>
            <PatientForm onSubmit={handleSubmit} loading={loading} />
          </div>

          {/* Results Column */}
          <div>
            {loading && (
              <div className="bg-white rounded-lg shadow-lg p-12 flex items-center justify-center">
                <div className="text-center">
                  <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-blue-600 mx-auto mb-4"></div>
                  <p className="text-gray-600">Analyzing patient data...</p>
                </div>
              </div>
            )}

            {!loading && result && <ResultDisplay result={result} />}

            {!loading && !result && !error && (
              <div className="bg-white rounded-lg shadow-lg p-12 flex items-center justify-center">
                <div className="text-center text-gray-500">
                  <svg
                    className="mx-auto h-16 w-16 mb-4"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"
                    />
                  </svg>
                  <p className="text-lg font-medium">No results yet</p>
                  <p className="text-sm mt-2">Fill in the patient information and click "Predict Risk"</p>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="mt-8 text-center text-sm text-gray-600">
          <p>
            Project for the Health Data Science Master's, Scientific Programming (Group D), DEPLOYMENT DEMO• For educational purposes only
          </p>
        </div>
      </div>
    </div>
  );
}

export default App;