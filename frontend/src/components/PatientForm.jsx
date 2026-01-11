import { useState } from 'react';

const PatientForm = ({ onSubmit, loading }) => {
  const [formData, setFormData] = useState({
    age: '',
    systolic_bp: '',
    diastolic_bp: '',
    bmi: '',
    heart_rate: '',
    total_cholesterol: '',
    glucose: '',
    cigarettes_per_day: '0',
    hypertension: '0'
  });

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    
    // Convert to numbers
    const data = {
      age: parseInt(formData.age),
      systolic_bp: parseFloat(formData.systolic_bp),
      diastolic_bp: parseFloat(formData.diastolic_bp),
      bmi: parseFloat(formData.bmi),
      heart_rate: parseFloat(formData.heart_rate),
      total_cholesterol: parseFloat(formData.total_cholesterol),
      glucose: parseFloat(formData.glucose),
      cigarettes_per_day: parseFloat(formData.cigarettes_per_day),
      hypertension: parseInt(formData.hypertension)
    };

    onSubmit(data);
  };

  return (
    <form onSubmit={handleSubmit} className="bg-white rounded-lg shadow-lg p-6 space-y-4">
      <h2 className="text-2xl font-bold text-gray-800 mb-4">Patient Information</h2>

      {/* Age */}
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-1">
          Age (years)
        </label>
        <input
          type="number"
          name="age"
          value={formData.age}
          onChange={handleChange}
          required
          min="18"
          max="120"
          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
      </div>

      {/* Blood Pressure */}
      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Systolic BP (mmHg)
          </label>
          <input
            type="number"
            name="systolic_bp"
            value={formData.systolic_bp}
            onChange={handleChange}
            required
            min="70"
            max="250"
            step="0.1"
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Diastolic BP (mmHg)
          </label>
          <input
            type="number"
            name="diastolic_bp"
            value={formData.diastolic_bp}
            onChange={handleChange}
            required
            min="40"
            max="150"
            step="0.1"
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>
      </div>

      {/* BMI and Heart Rate */}
      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            BMI (kg/m²)
          </label>
          <input
            type="number"
            name="bmi"
            value={formData.bmi}
            onChange={handleChange}
            required
            min="15"
            max="60"
            step="0.1"
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Heart Rate (bpm)
          </label>
          <input
            type="number"
            name="heart_rate"
            value={formData.heart_rate}
            onChange={handleChange}
            required
            min="40"
            max="200"
            step="0.1"
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>
      </div>

      {/* Cholesterol and Glucose */}
      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Total Cholesterol (mg/dL)
          </label>
          <input
            type="number"
            name="total_cholesterol"
            value={formData.total_cholesterol}
            onChange={handleChange}
            required
            min="100"
            max="500"
            step="0.1"
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Glucose (mg/dL)
          </label>
          <input
            type="number"
            name="glucose"
            value={formData.glucose}
            onChange={handleChange}
            required
            min="50"
            max="400"
            step="0.1"
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>
      </div>

      {/* Cigarettes per day */}
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-1">
          Cigarettes per Day
        </label>
        <input
          type="number"
          name="cigarettes_per_day"
          value={formData.cigarettes_per_day}
          onChange={handleChange}
          required
          min="0"
          step="0.1"
          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
      </div>

      {/* Hypertension */}
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-1">
          Hypertension Diagnosis
        </label>
        <select
          name="hypertension"
          value={formData.hypertension}
          onChange={handleChange}
          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
        >
          <option value="0">No</option>
          <option value="1">Yes</option>
        </select>
      </div>

      {/* Submit Button */}
      <button
        type="submit"
        disabled={loading}
        className={`w-full py-3 px-4 rounded-md text-white font-medium transition-colors ${
          loading
            ? 'bg-gray-400 cursor-not-allowed'
            : 'bg-blue-600 hover:bg-blue-700'
        }`}
      >
        {loading ? 'Analyzing...' : 'Predict Risk'}
      </button>
    </form>
  );
};

export default PatientForm;