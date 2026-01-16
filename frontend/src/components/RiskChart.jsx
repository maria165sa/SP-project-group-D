import { PieChart, Pie, Cell, ResponsiveContainer, Legend } from 'recharts';

const RiskChart = ({ probability, riskLevel }) => {
  const data = [
    { name: 'Risk', value: probability * 100 },
    { name: 'No Risk', value: (1 - probability) * 100 }
  ];

  const COLORS = {
    Low: ['#10b981', '#e5e7eb'],
    Medium: ['#f59e0b', '#e5e7eb'],
    High: ['#ef4444', '#e5e7eb']
  };

  const colors = COLORS[riskLevel] || COLORS.Medium;

  return (
    <div className="w-full h-64">
      <ResponsiveContainer width="100%" height="100%">
        <PieChart>
          <Pie
            data={data}
            cx="50%"
            cy="50%"
            labelLine={false}
            label={({ name, value }) => `${name}: ${value.toFixed(1)}%`}
            outerRadius={80}
            fill="#8884d8"
            dataKey="value"
          >
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={colors[index]} />
            ))}
          </Pie>
          <Legend />
        </PieChart>
      </ResponsiveContainer>
    </div>
  );
};

export default RiskChart;