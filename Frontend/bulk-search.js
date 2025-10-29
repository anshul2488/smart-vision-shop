/* Grocery OCR Pipeline Frontend */
(function () {
  const textArea = document.getElementById('groceryText');
  const parseTextBtn = document.getElementById('parseTextBtn');
  const clearTextBtn = document.getElementById('clearTextBtn');
  const parseImageBtn = document.getElementById('parseImageBtn');
  const compareBtn = document.getElementById('compareBtn');
  const sampleBtn = document.getElementById('sampleBtn');
  const itemsList = document.getElementById('itemsList');
  const platformCards = document.getElementById('platformCards');
  const compareStatus = document.getElementById('compareStatus');
  const imageInput = document.getElementById('imageInput');
  const toast = document.getElementById('toast');
  const resetBtn = document.getElementById('resetBtn');

  const state = {
    items: [],
    platforms: ['Amazon', 'Zepto', 'BigBasket', 'JioMart'],
    priceData: null,
  };

  // API base URL
  const API_BASE = 'http://localhost:5000/api';

  document.getElementById('year').textContent = String(new Date().getFullYear());

  function showToast(message, timeoutMs = 2000) {
    toast.textContent = message;
    toast.hidden = false;
    window.clearTimeout(showToast._t);
    showToast._t = window.setTimeout(() => (toast.hidden = true), timeoutMs);
  }

  function sanitize(text) {
    return text.replace(/[<>]/g, '');
  }

  // API functions
  async function parseTextAPI(text) {
    try {
      const response = await fetch(`${API_BASE}/parse-text`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text }),
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Error parsing text:', error);
      throw error;
    }
  }

  async function parseImageAPI(imageFile) {
    try {
      const formData = new FormData();
      formData.append('image', imageFile);
      
      const response = await fetch(`${API_BASE}/parse-image`, {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Error parsing image:', error);
      throw error;
    }
  }

  async function searchPricesAPI(items, platforms = ['amazon', 'zepto', 'bigbasket', 'jiomart']) {
    try {
      const response = await fetch(`${API_BASE}/search-prices`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ items, platforms }),
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Error searching prices:', error);
      throw error;
    }
  }

  async function getSampleDataAPI() {
    try {
      const response = await fetch(`${API_BASE}/sample-data`);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Error getting sample data:', error);
      throw error;
    }
  }

  function parseTextToItems(text) {
    const lines = text
      .split(/\r?\n/)
      .map((l) => l.trim())
      .filter(Boolean);
    const result = [];
    for (const line of lines) {
      const quantityMatch = line.match(/(^|\s)(\d+\s?[a-zA-Z]*)$/);
      const numberMatch = line.match(/\b(\d+)\b/);
      const quantity = quantityMatch ? quantityMatch[2] : numberMatch ? numberMatch[1] : '1';
      const name = sanitize(line.replace(quantityMatch?.[2] || numberMatch?.[1] || '', '').trim()) || sanitize(line);
      result.push({ 
        name, 
        quantity: String(quantity).trim() || '1',
        unit: 'pieces' // Default unit
      });
    }
    return result;
  }

  function renderItems() {
    itemsList.innerHTML = '';
    if (state.items.length === 0) {
      const li = document.createElement('li');
      li.className = 'muted';
      li.textContent = 'No items yet. Add some above.';
      itemsList.appendChild(li);
      compareBtn.disabled = true;
      return;
    }
    compareBtn.disabled = false;
    for (const [index, item] of state.items.entries()) {
      const li = document.createElement('li');
      li.className = 'item-row';
      li.innerHTML = `
        <span class="item-name">${item.name || item.item_name}</span>
        <span class="item-qty">× ${item.quantity} ${item.unit || ''}</span>
        ${item.confidence ? `<span class="confidence">(${Math.round(item.confidence * 100)}%)</span>` : ''}
      `;
      li.title = `Click to remove ${item.name || item.item_name}`;
      li.addEventListener('click', () => {
        state.items.splice(index, 1);
        renderItems();
        renderCards(null);
      });
      itemsList.appendChild(li);
    }
  }

  function processPriceData(apiResults) {
    const data = {};
    
    // Process each platform
    const platformNames = {
      'amazon': 'Amazon',
      'zepto': 'Zepto',
      'bigbasket': 'BigBasket',
      'jiomart': 'JioMart'
    };
    
    for (const result of apiResults) {
      const itemName = result.item_name;
      const quantity = result.quantity;
      const unit = result.unit;
      
      // Process each platform for this item
      for (const [platformKey, platformName] of Object.entries(platformNames)) {
        if (!data[platformName]) {
          data[platformName] = { items: [], delivery: 0, total: 0 };
        }
        
        const platformData = result.platforms[platformKey];
        if (platformData && platformData.products && platformData.products.length > 0) {
          // Use the first (best) product
          const product = platformData.products[0];
          const priceText = product.price || 'N/A';
          // Use calculated total if available, otherwise parse price
          const price = product.calculated_total || parseFloat(priceText.replace(/[^\d.]/g, '')) || 0;
          
          // Check if this item already exists for this platform
          const existingItemIndex = data[platformName].items.findIndex(item => item.name === itemName);
          if (existingItemIndex >= 0) {
            // Update existing item
            data[platformName].items[existingItemIndex] = {
              name: itemName,
              quantity: `${quantity} ${unit}`,
              price: price,
              unit_price: product.unit_price || 0,
              product_url: product.product_url,
              rating: product.rating,
              brand: product.brand || '',
              variant: product.variant || '',
              eta: product.eta || '',
              match_score: product.match_score || 0,
              total_products_found: product.total_products_found || 0
            };
          } else {
            // Add new item
            data[platformName].items.push({
              name: itemName,
              quantity: `${quantity} ${unit}`,
              price: price,
              unit_price: product.unit_price || 0,
              product_url: product.product_url,
              rating: product.rating,
              brand: product.brand || '',
              variant: product.variant || '',
              eta: product.eta || '',
              match_score: product.match_score || 0,
              total_products_found: product.total_products_found || 0
            });
          }
        } else {
          // No products found for this platform
          const existingItemIndex = data[platformName].items.findIndex(item => item.name === itemName);
          if (existingItemIndex < 0) {
            data[platformName].items.push({
              name: itemName,
              quantity: `${quantity} ${unit}`,
              price: 0,
              product_url: '',
              rating: 'N/A',
              brand: '',
              variant: '',
              eta: ''
            });
          }
        }
      }
    }
    
    // Calculate totals and delivery for each platform
    for (const [platformName, platformData] of Object.entries(data)) {
      const subtotal = platformData.items.reduce((sum, item) => sum + item.price, 0);
      const delivery = (platformName === 'Zepto' || platformName === 'BigBasket' || platformName === 'JioMart') ? 0 : 50; // Zepto, BigBasket, and JioMart have free delivery
      platformData.delivery = delivery;
      platformData.total = subtotal + delivery;
    }
    
    return data;
  }

  function renderLoadingCards() {
    platformCards.innerHTML = '';
    for (const platform of state.platforms) {
      const card = document.createElement('div');
      card.className = 'card loading';
      card.innerHTML = `
        <div class="card-header">
          <div class="row gap-sm">
            <h3 style="margin:0">${platform}</h3>
          </div>
          <div class="price-total">Loading...</div>
        </div>
        <div class="products-container">
          ${state.items.map(() => `
            <div class="product-card">
              <div class="product-info">
                <div class="product-name">Loading...</div>
                <div class="product-details">
                  <div class="loading-spinner"></div>
                </div>
              </div>
              <div class="price-info">
                <div class="total-price">...</div>
              </div>
            </div>
          `).join('')}
        </div>
      `;
      platformCards.appendChild(card);
    }
  }

  function renderCards(priceData) {
    platformCards.innerHTML = '';
    state.priceData = priceData;
    if (!state.items.length) {
      compareStatus.textContent = 'Add items above, then compare.';
      return;
    }
    if (!priceData) {
      compareStatus.textContent = 'Ready to compare across platforms.';
      return;
    }
    
    const totals = Object.values(priceData).map((p) => p.total);
    const minTotal = Math.min(...totals);
    compareStatus.textContent = 'Real prices from Amazon India, Zepto, BigBasket, and JioMart.';

    // Create comparison table
    const table = document.createElement('div');
    table.className = 'comparison-table';
    
    // Get all unique items across platforms
    const allItems = new Set();
    Object.values(priceData).forEach(platformData => {
      platformData.items.forEach(item => {
        allItems.add(item.name);
      });
    });
    
    const platforms = Object.keys(priceData);
    
    // Create table header
    let tableHTML = `
      <div class="table-header">
        <div class="table-row">
          <div class="item-column">Item</div>
          ${platforms.map(platform => `<div class="platform-column">${platform}</div>`).join('')}
          <div class="best-column">Best Price</div>
        </div>
      </div>
      <div class="table-body">
    `;
    
    // Create rows for each item
    Array.from(allItems).forEach(itemName => {
      const itemPrices = {};
      const itemDetails = {};
      
      // Collect prices and details for this item across all platforms
      platforms.forEach(platform => {
        const item = priceData[platform].items.find(i => i.name === itemName);
        if (item) {
          itemPrices[platform] = item.price;
          itemDetails[platform] = item;
        } else {
          itemPrices[platform] = null;
          itemDetails[platform] = null;
        }
      });
      
      // Find the cheapest price
      const validPrices = Object.values(itemPrices).filter(p => p !== null && p > 0);
      const minPrice = validPrices.length > 0 ? Math.min(...validPrices) : null;
      const cheapestPlatform = minPrice ? Object.keys(itemPrices).find(p => itemPrices[p] === minPrice) : null;
      
      // Get item details (use first available platform for item info)
      const firstAvailableItem = Object.values(itemDetails).find(item => item !== null);
      const itemQuantity = firstAvailableItem ? firstAvailableItem.quantity : '';
      
      tableHTML += `
        <div class="table-row item-row">
          <div class="item-column">
            <div class="item-name">${sanitize(itemName)}</div>
            <div class="item-quantity">${sanitize(itemQuantity)}</div>
          </div>
          ${platforms.map(platform => {
            const item = itemDetails[platform];
            const price = itemPrices[platform];
            const isCheapest = platform === cheapestPlatform && price === minPrice;
            
            if (item && price > 0) {
              return `
                <div class="platform-column ${isCheapest ? 'cheapest' : ''}">
                  <div class="price-cell">
                    <div class="price">₹ ${price.toLocaleString('en-IN')}</div>
                    ${item.rating && item.rating !== 'N/A' ? `<div class="rating">${item.rating}</div>` : ''}
                    ${item.product_url ? `<a href="${item.product_url}" target="_blank" class="product-link">View</a>` : ''}
                  </div>
                </div>
              `;
            } else {
              return `
                <div class="platform-column unavailable">
                  <div class="price-cell">
                    <div class="price">N/A</div>
                    <div class="unavailable-text">Not available</div>
                  </div>
                </div>
              `;
            }
          }).join('')}
          <div class="best-column">
            ${cheapestPlatform ? `
              <div class="best-price">
                <div class="best-platform">${cheapestPlatform}</div>
                <div class="best-amount">₹ ${minPrice.toLocaleString('en-IN')}</div>
              </div>
            ` : '<div class="no-best">N/A</div>'}
          </div>
        </div>
      `;
    });
    
    // Add totals row
    tableHTML += `
        <div class="table-row totals-row">
          <div class="item-column">
            <div class="total-label">TOTAL</div>
            <div class="delivery-note">+ Delivery charges</div>
          </div>
          ${platforms.map(platform => {
            const info = priceData[platform];
            const isCheapest = info.total === minTotal;
            return `
              <div class="platform-column ${isCheapest ? 'cheapest' : ''}">
                <div class="price-cell">
                  <div class="price total-price">₹ ${info.total.toLocaleString('en-IN')}</div>
                  <div class="delivery">Delivery: ₹ ${info.delivery.toLocaleString('en-IN')}</div>
                  ${isCheapest ? '<div class="cheapest-badge">Cheapest</div>' : ''}
                </div>
              </div>
            `;
          }).join('')}
          <div class="best-column">
            <div class="best-price">
              <div class="best-platform">${Object.keys(priceData).find(p => priceData[p].total === minTotal)}</div>
              <div class="best-amount">₹ ${minTotal.toLocaleString('en-IN')}</div>
            </div>
          </div>
        </div>
      </div>
    `;
    
    table.innerHTML = tableHTML;
    platformCards.appendChild(table);
    
    // Add platform action buttons below the table
    const buttonsContainer = document.createElement('div');
    buttonsContainer.className = 'platform-buttons';
    buttonsContainer.innerHTML = platforms.map(platform => `
      <button class="btn platform-btn" data-platform="${platform}">View on ${platform}</button>
    `).join('');
    
    // Add event listeners to buttons
    buttonsContainer.querySelectorAll('.platform-btn').forEach(button => {
      button.addEventListener('click', () => redirectToCart(button.dataset.platform));
    });
    
    platformCards.appendChild(buttonsContainer);
  }

  function redirectToCart(platform) {
    if (!state.items.length) return;
    
    let url;
    if (platform.toLowerCase() === 'zepto') {
      // For Zepto, redirect to the main site
      url = 'https://www.zeptonow.com/';
    } else if (platform.toLowerCase() === 'bigbasket') {
      // For BigBasket, redirect to the main site
      url = 'https://www.bigbasket.com/';
    } else if (platform.toLowerCase() === 'jiomart') {
      // For JioMart, redirect to the main site
      url = 'https://www.jiomart.com/';
    } else {
      // For Amazon and other platforms
      const base = platform.toLowerCase();
      const params = new URLSearchParams();
      params.set('items', JSON.stringify(state.items));
      url = `https://www.${base}.in/search?` + params.toString();
    }
    
    showToast(`Opening ${platform}...`);
    window.open(url, '_blank');
  }

  async function handleParseText() {
    try {
      const text = textArea.value.trim();
      if (!text) {
        showToast('Please enter some text first.');
        return;
      }

      showToast('Parsing text...');
      parseTextBtn.disabled = true;
      
      const result = await parseTextAPI(text);
      
      if (result.success) {
        state.items = result.items;
        renderItems();
        renderCards(null);
        showToast(`Parsed ${result.total_items} items successfully!`);
      } else {
        showToast('Failed to parse text. Please try again.');
      }
    } catch (error) {
      console.error('Error parsing text:', error);
      showToast('Error parsing text. Please try again.');
    } finally {
      parseTextBtn.disabled = false;
    }
  }

  async function handleParseImage() {
    try {
      if (!imageInput.files || imageInput.files.length === 0) {
        showToast('Choose an image first.');
        return;
      }

      showToast('Parsing image...');
      parseImageBtn.disabled = true;
      
      const result = await parseImageAPI(imageInput.files[0]);
      
      if (result.success) {
        state.items = result.items;
        renderItems();
        renderCards(null);
        
        // Show detailed processing information
        let message = `Parsed ${result.total_items} items from image!`;
        if (result.processing_info) {
          message += ` (OCR confidence: ${Math.round(result.ocr_confidence * 100)}%)`;
        }
        showToast(message);
        
        // Log processing details to console
        if (result.processing_info) {
          console.log('Processing details:', result.processing_info);
        }
      } else {
        showToast('Failed to parse image. Please try again.');
      }
    } catch (error) {
      console.error('Error parsing image:', error);
      showToast('Error parsing image. Please try again.');
    } finally {
      parseImageBtn.disabled = false;
    }
  }

  async function handleCompare() {
    if (!state.items.length) return;
    
    try {
      // Enhanced progress tracking
      let progressStep = 0;
      const totalSteps = 4;
      
      const updateProgress = (step, message) => {
        progressStep = step;
        const percentage = Math.round((step / totalSteps) * 100);
        compareStatus.innerHTML = `
          <div class="loading-text">
            <div class="loading-spinner"></div>
            ${message} (${percentage}%)
          </div>
        `;
      };
      
      updateProgress(1, 'Initializing price search...');
      compareBtn.disabled = true;
      
      // Show loading cards
      renderLoadingCards();
      
      updateProgress(2, 'Searching Amazon...');
      await new Promise(resolve => setTimeout(resolve, 100)); // Small delay for visual feedback
      
      updateProgress(3, 'Searching other platforms...');
      await new Promise(resolve => setTimeout(resolve, 100));
      
      updateProgress(4, 'Processing results...');
      const result = await searchPricesAPI(state.items);
      
      if (result.success) {
        const priceData = processPriceData(result.results);
        renderCards(priceData);
        showToast(`Found prices for ${result.total_items} items!`);
      } else {
        showToast('Failed to search prices. Please try again.');
        renderCards(null);
      }
    } catch (error) {
      console.error('Error searching prices:', error);
      showToast('Error searching prices. Please try again.');
      renderCards(null);
    } finally {
      compareBtn.disabled = false;
    }
  }

  async function handleSample() {
    try {
      showToast('Loading sample data...');
      sampleBtn.disabled = true;
      
      const result = await getSampleDataAPI();
      
      if (result.success) {
        state.items = result.items;
        renderItems();
        renderCards(null);
        showToast('Sample data loaded!');
      } else {
        showToast('Failed to load sample data.');
      }
    } catch (error) {
      console.error('Error loading sample data:', error);
      showToast('Error loading sample data.');
    } finally {
      sampleBtn.disabled = false;
    }
  }

  function handleReset() {
    textArea.value = '';
    imageInput.value = '';
    state.items = [];
    state.priceData = null;
    renderItems();
    renderCards(null);
    showToast('Cleared.');
  }

  parseTextBtn.addEventListener('click', handleParseText);
  clearTextBtn.addEventListener('click', handleReset);
  parseImageBtn.addEventListener('click', handleParseImage);
  compareBtn.addEventListener('click', handleCompare);
  sampleBtn.addEventListener('click', handleSample);
  resetBtn.addEventListener('click', handleReset);

  renderItems();
  renderCards(null);
})();