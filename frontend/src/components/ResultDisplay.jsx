import RiskChart from './RiskChart';

const ResultDisplay = ({ result }) => {
  if (!result) return null;

  const getRiskColor = (level) => {
    switch (level) {
      case 'Low':
        return 'bg-green-100 text-green-800 border-green-300';
      case 'Medium':
        return 'bg-yellow-100 text-yellow-800 border-yellow-300';
      case 'High':
        return 'bg-red-100 text-red-800 border-red-300';
      default:
        return 'bg-gray-100 text-gray-800 border-gray-300';
    }
  };

  const getRiskIcon = (level) => {
    switch (level) {
      case 'Low':
        return '✓';
      case 'Medium':
        return '⚠';
      case 'High':
        return '⚠';
      default:
        return 'ℹ';
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-lg p-6 space-y-6">
      <h2 className="text-2xl font-bold text-gray-800">Prediction Result</h2>
      
      {/* Risk Level Badge */}
      <div className={`rounded-lg border-2 p-4 ${getRiskColor(result.risk_level)}`}>
        <div className="flex items-center justify-center space-x-3">
          <span className="text-3xl">{getRiskIcon(result.risk_level)}</span>
          <div>
            <p className="text-sm font-medium">Risk Level</p>
            <p className="text-2xl font-bold">{result.risk_level}</p>
          </div>
        </div>
      </div>

      {/* Probability */}
      <div className="bg-gray-50 rounded-lg p-4">
        <p className="text-sm text-gray-600 mb-1">10-Year CHD Risk Probability</p>
        <p className="text-3xl font-bold text-gray-900">
          {(result.probability * 100).toFixed(1)}%
        </p>
        <div className="mt-3 bg-gray-200 rounded-full h-3 overflow-hidden">
          <div
            className={`h-full transition-all duration-500 ${
              result.risk_level === 'Low' ? 'bg-green-500' :
              result.risk_level === 'Medium' ? 'bg-yellow-500' : 'bg-red-500'
            }`}
            style={{ width: `${result.probability * 100}%` }}
          />
        </div>
      </div>

      {/* Chart */}
      <div>
        <h3 className="text-lg font-semibold text-gray-700 mb-3">Risk Distribution</h3>
        <RiskChart probability={result.probability} riskLevel={result.risk_level} />
      </div>

      {/* Prediction Label */}
      <div className="border-t pt-4">
        <p className="text-sm text-gray-600">Prediction</p>
        <p className="text-lg font-medium text-gray-800">{result.prediction_label}</p>
      </div>
    </div>
  );
};

export default ResultDisplay;