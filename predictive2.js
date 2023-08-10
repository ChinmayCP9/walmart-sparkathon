const fs = require('fs');

// Read the JSON file
fs.readFile('products1.json', 'utf8', (err, data) => {
  if (err) {
    console.error('Error reading JSON file:', err);
    return;
  }

  try {
    const dataset = JSON.parse(data);

    // Calculate correlation coefficient
    const correlation = calculateCorrelation(dataset, 'rating', 'positiveReviews');
    console.log('Correlation coefficient:', correlation);
  } catch (err) {
    console.error('Error parsing JSON:', err);
  }
});

// Calculate correlation coefficient
function calculateCorrelation(data, xKey, yKey) {
  // Implementation of the calculateCorrelation function goes here
}
