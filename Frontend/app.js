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
    platforms: ['Amazon', 'Blinkit', 'Zepto', 'BigBasket'],
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

  async function searchPricesAPI(items, platforms = ['amazon', 'blinkit', 'zepto', 'bigbasket']) {
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
      'blinkit': 'Blinkit',
      'zepto': 'Zepto',
      'bigbasket': 'BigBasket'
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
      const delivery = (platformName === 'Blinkit' || platformName === 'Zepto' || platformName === 'BigBasket') ? 0 : 50; // Blinkit, Zepto, and BigBasket have free delivery
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
    compareStatus.textContent = 'Real prices from Amazon India, Blinkit, Zepto, and BigBasket.';

    for (const [index, platform] of Object.keys(priceData).entries()) {
      const info = priceData[platform];
      const card = document.createElement('div');
      const isBest = info.total === minTotal;

      card.className = 'card' + (isBest ? ' best' : '');
      card.style.animationDelay = `${index * 0.1}s`;
      card.innerHTML = `
        <div class="card-header">
          <div class="row gap-sm">
            <h3 style="margin:0">${platform}</h3>
            ${isBest ? '<span class="badge badge--best">Cheapest</span>' : ''}
          </div>
          <div class="price-total${isBest ? ' best' : ''}">₹ ${info.total.toLocaleString('en-IN')}</div>
        </div>
        <div class="products-container">
          ${info.items
            .map((it) => `
              <div class="product-card">
                <div class="product-info">
                  <div class="product-name">
                    ${sanitize(it.name)} × ${sanitize(String(it.quantity))}
                    ${it.unit_price > 0 ? `<span class="price-info-icon" title="Price calculation: Unit price × Quantity = Total price">ℹ️</span>` : ''}
                  </div>
                  <div class="product-details">
                    ${it.brand ? `<div class="product-detail-item"><span class="icon">🏷️</span>${sanitize(it.brand)}</div>` : ''}
                    ${it.variant ? `<div class="product-detail-item"><span class="icon">📦</span>${sanitize(it.variant)}</div>` : ''}
                    ${it.eta ? `<div class="product-detail-item"><span class="icon">🚚</span>${sanitize(it.eta)}</div>` : ''}
                    ${it.unit_price > 0 ? `<div class="product-detail-item"><span class="icon">💰</span>₹${it.unit_price.toFixed(2)}/${it.quantity.split(' ')[1] || 'unit'}</div>` : ''}
                    <div class="match-score">Match: ${it.match_score || 0}/20</div>
                    <div class="products-found">Found: ${it.total_products_found || 0} products</div>
                  </div>
                  ${it.product_url ? `<a href="${it.product_url}" target="_blank" class="product-link">View Product</a>` : ''}
                </div>
                <div class="price-info">
                  <div class="total-price">₹ ${it.price.toLocaleString('en-IN')}</div>
                  ${it.unit_price > 0 ? `<div class="price-breakdown">🧮 ₹${it.unit_price.toFixed(2)} × ${it.quantity.split(' ')[0] || '1'} = ₹${it.price.toLocaleString('en-IN')}</div>` : ''}
                  ${it.rating && it.rating !== 'N/A' ? `<div class="rating-stars">${it.rating}</div>` : '<div class="muted">No rating</div>'}
                </div>
              </div>
            `)
            .join('')}
          <div class="delivery-info">
            <div class="product-detail-item">
              <span class="icon">🚚</span>Delivery
            </div>
            <div class="price-info">
              <div class="total-price">₹ ${info.delivery.toLocaleString('en-IN')}</div>
            </div>
          </div>
        </div>
        <button class="btn" data-platform="${platform}">View on ${platform}</button>
      `;
      const button = card.querySelector('button');
      button.addEventListener('click', () => redirectToCart(platform));
      platformCards.appendChild(card);
    }
  }

  function redirectToCart(platform) {
    if (!state.items.length) return;
    
    let url;
    if (platform.toLowerCase() === 'blinkit') {
      // For Blinkit, redirect to the main site
      url = 'https://blinkit.com/';
    } else if (platform.toLowerCase() === 'zepto') {
      // For Zepto, redirect to the main site
      url = 'https://www.zeptonow.com/';
    } else if (platform.toLowerCase() === 'bigbasket') {
      // For BigBasket, redirect to the main site
      url = 'https://www.bigbasket.com/';
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
      compareStatus.innerHTML = '<div class="loading-text"><div class="loading-spinner"></div>Searching for prices...</div>';
      compareBtn.disabled = true;
      
      // Show loading cards
      renderLoadingCards();
      
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