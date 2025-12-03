/**
 * Number Formatting Utilities
 * Consistent number display across the application
 */

export function formatNumber(num, decimals = 4) {
    if (num === null || num === undefined || isNaN(num)) return 'N/A';
    return Number(num).toFixed(decimals);
}

export function formatPercentage(num, decimals = 2) {
    if (num === null || num === undefined || isNaN(num)) return 'N/A';
    return `${(num * 100).toFixed(decimals)}%`;
}

export function formatScientific(num, decimals = 2) {
    if (num === null || num === undefined || isNaN(num)) return 'N/A';
    return num.toExponential(decimals);
}

export function formatArray(arr, decimals = 4) {
    if (!Array.isArray(arr)) return 'N/A';
    return arr.map(num => formatNumber(num, decimals)).join(', ');
}

export function formatMatrix(matrix, decimals = 4) {
    if (!Array.isArray(matrix)) return 'N/A';
    return matrix.map(row => formatArray(row, decimals)).join('\\n');
}
