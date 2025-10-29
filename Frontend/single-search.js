/* SmartVisionShop Single Search */
(function () {
  // Single search elements
  const singleItemInput = document.getElementById('singleItemInput');
  const searchSingleBtn = document.getElementById('searchSingleBtn');
  const autocompleteDropdown = document.getElementById('autocompleteDropdown');
  const relatedItemsSection = document.getElementById('relatedItemsSection');
  const relatedItems = document.getElementById('relatedItems');
  const platformCards = document.getElementById('platformCards');
  const compareStatus = document.getElementById('compareStatus');
  const toast = document.getElementById('toast');
  const resetBtn = document.getElementById('resetBtn');

  const state = {
    items: [],
    platforms: ['Amazon', 'Zepto', 'BigBasket', 'JioMart'],
    priceData: null,
  };

  // Autocomplete data for grocery items
  const autocompleteData = [
    { name: 'milk', icon: '🥛', category: 'Dairy' },
    { name: 'bread', icon: '🍞', category: 'Bakery' },
    { name: 'eggs', icon: '🥚', category: 'Dairy' },
    { name: 'rice', icon: '🍚', category: 'Grains' },
    { name: 'tomatoes', icon: '🍅', category: 'Vegetables' },
    { name: 'onions', icon: '🧅', category: 'Vegetables' },
    { name: 'potatoes', icon: '🥔', category: 'Vegetables' },
    { name: 'butter', icon: '🧈', category: 'Dairy' },
    { name: 'cheese', icon: '🧀', category: 'Dairy' },
    { name: 'yogurt', icon: '🥛', category: 'Dairy' },
    { name: 'chicken', icon: '🍗', category: 'Meat' },
    { name: 'fish', icon: '🐟', category: 'Seafood' },
    { name: 'bananas', icon: '🍌', category: 'Fruits' },
    { name: 'apples', icon: '🍎', category: 'Fruits' },
    { name: 'oranges', icon: '🍊', category: 'Fruits' },
    { name: 'carrots', icon: '🥕', category: 'Vegetables' },
    { name: 'spinach', icon: '🥬', category: 'Vegetables' },
    { name: 'cucumber', icon: '🥒', category: 'Vegetables' },
    { name: 'garlic', icon: '🧄', category: 'Vegetables' },
    { name: 'ginger', icon: '🫚', category: 'Vegetables' },
    { name: 'oil', icon: '🫒', category: 'Cooking' },
    { name: 'salt', icon: '🧂', category: 'Spices' },
    { name: 'sugar', icon: '🍯', category: 'Sweeteners' },
    { name: 'flour', icon: '🌾', category: 'Grains' },
    { name: 'pasta', icon: '🍝', category: 'Grains' },
    { name: 'cereal', icon: '🥣', category: 'Breakfast' },
    { name: 'coffee', icon: '☕', category: 'Beverages' },
    { name: 'tea', icon: '🍵', category: 'Beverages' },
    { name: 'juice', icon: '🧃', category: 'Beverages' },
    { name: 'water', icon: '💧', category: 'Beverages' },
    { name: 'cookies', icon: '🍪', category: 'Snacks' },
    { name: 'chips', icon: '🍟', category: 'Snacks' },
    { name: 'nuts', icon: '🥜', category: 'Snacks' },
    { name: 'chocolate', icon: '🍫', category: 'Sweets' },
    { name: 'ice cream', icon: '🍦', category: 'Desserts' },
  ];

  // Related items mapping
  const relatedItemsMap = {
    'milk': ['butter', 'cheese', 'yogurt', 'cereal'],
    'bread': ['butter', 'jam', 'eggs', 'cheese'],
    'eggs': ['milk', 'bread', 'butter', 'cheese'],
    'rice': ['oil', 'onions', 'garlic', 'vegetables'],
    'tomatoes': ['onions', 'garlic', 'oil', 'basil'],
    'onions': ['tomatoes', 'garlic', 'potatoes', 'oil'],
    'potatoes': ['onions', 'oil', 'butter', 'salt'],
    'butter': ['bread', 'milk', 'eggs', 'cheese'],
    'chicken': ['onions', 'garlic', 'oil', 'spices'],
    'fish': ['lemon', 'oil', 'garlic', 'herbs'],
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

  function clearAllData() {
    state.items = [];
    state.priceData = null;
    singleItemInput.value = '';
    hideAutocomplete();
    hideRelatedItems();
    platformCards.className = 'cards'; // Reset to default vertical layout
    renderCards(null);
  }

  // Autocomplete functions
  function showAutocomplete(query) {
    if (!query || query.length < 1) {
      hideAutocomplete();
      return;
    }

    const matches = autocompleteData.filter(item => 
      item.name.toLowerCase().includes(query.toLowerCase())
    ).slice(0, 8);

    if (matches.length === 0) {
      hideAutocomplete();
      return;
    }

    autocompleteDropdown.innerHTML = matches.map(item => `
      <div class="autocomplete-item" data-item="${item.name}">
        <span class="icon">${item.icon}</span>
        <span class="text">${item.name}</span>
        <span class="subtext">${item.category}</span>
      </div>
    `).join('');

    autocompleteDropdown.style.display = 'block';

    // Add click handlers
    autocompleteDropdown.querySelectorAll('.autocomplete-item').forEach(item => {
      item.addEventListener('click', () => {
        const itemName = item.dataset.item;
        singleItemInput.value = itemName;
        hideAutocomplete();
        updateSearchButton();
      });
    });
  }

  function hideAutocomplete() {
    autocompleteDropdown.style.display = 'none';
  }

  function updateSearchButton() {
    const hasValue = singleItemInput.value.trim().length > 0;
    searchSingleBtn.disabled = !hasValue;
  }

  // Related items functions
  function showRelatedItems(searchItem) {
    const related = relatedItemsMap[searchItem.toLowerCase()];
    if (!related || related.length === 0) {
      hideRelatedItems();
      return;
    }

    const relatedData = related.map(itemName => {
      const item = autocompleteData.find(i => i.name === itemName);
      return item || { name: itemName, icon: '🛒', category: 'Grocery' };
    });

    relatedItems.innerHTML = relatedData.map(item => `
      <div class="related-item-card" data-item="${item.name}">
        <div class="icon">${item.icon}</div>
        <div class="name">${item.name}</div>
        <div class="price">Search prices</div>
      </div>
    `).join('');

    relatedItemsSection.style.display = 'block';

    // Add click handlers for related items
    relatedItems.querySelectorAll('.related-item-card').forEach(card => {
      card.addEventListener('click', () => {
        const itemName = card.dataset.item;
        singleItemInput.value = itemName;
        updateSearchButton();
        handleSingleSearch();
      });
    });
  }

  function hideRelatedItems() {
    relatedItemsSection.style.display = 'none';
  }

  function sanitize(text) {
    return text.replace(/[<>]/g, '');
  }

  // API functions
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
          // Process all products for this platform
          const allProducts = platformData.products;
          console.log(`${platformName} has ${allProducts.length} products:`, allProducts);
          
          // Add all products to the platform data
          for (const product of allProducts) {
            // Simply use the raw price without complex calculations
            const price = typeof product.price === 'number' ? product.price : parseFloat(product.price) || 0;
            
            // Add product to platform data
            data[platformName].items.push({
              name: itemName,
              quantity: `${quantity} ${unit}`,
              price: price,
              product_url: product.product_url,
              rating: product.rating,
              brand: product.brand || '',
              variant: product.variant || '',
              eta: product.eta || '',
              match_score: product.match_score || 0,
              total_products_found: product.total_products_found || 0,
              product_name: product.product_name || product.name || itemName,
              image_url: product.image_url || '',
              review_count: product.review_count || '',
              inventory: product.inventory || 0
            });
            console.log(`Added product to ${platformName}:`, product.name, 'Total items now:', data[platformName].items.length);
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
    
    // For single search, we don't need complex totals - just show individual products
    for (const [platformName, platformData] of Object.entries(data)) {
      platformData.delivery = 0; // No delivery charges shown in single search
      platformData.total = 0; // No total calculation needed
      console.log(`Final ${platformName} data:`, platformData.items.length, 'items');
    }
    
    console.log('Final processed data:', data);
    return data;
  }

  function renderLoadingCards() {
    platformCards.innerHTML = '';
    platformCards.className = 'cards'; // Remove horizontal class - platforms should be vertical
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
        <div class="products-container scrollable-products">
          ${Array(3).fill().map(() => `
            <div class="product-card">
              <div class="product-image">
                <div class="no-image">⏳</div>
              </div>
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
    console.log('Rendering cards with data:', priceData);
    platformCards.innerHTML = '';
    platformCards.className = 'cards'; // Remove horizontal class - platforms should be vertical
    state.priceData = priceData;
    if (!priceData) {
      compareStatus.textContent = 'Search for items above to compare prices.';
      return;
    }
    
    compareStatus.textContent = 'Real prices from Amazon India, Zepto, BigBasket, and JioMart.';

    for (const [index, platform] of Object.keys(priceData).entries()) {
      const info = priceData[platform];
      console.log(`Rendering ${platform} with ${info.items.length} items:`, info.items);
      const card = document.createElement('div');

      card.className = 'card';
      card.style.animationDelay = `${index * 0.1}s`;
      card.innerHTML = `
        <div class="card-header">
          <div class="row gap-sm">
            <h3 style="margin:0">${platform}</h3>
          </div>
        </div>
        <div class="products-container scrollable-products">
          ${info.items
            .map((it, index) => `
              <div class="product-card" style="animation-delay: ${index * 0.1}s">
                <div class="product-image">
                  ${it.image_url ? `<img src="${it.image_url}" alt="${sanitize(it.product_name || it.name)}" class="product-img" onerror="this.style.display='none'">` : '<div class="no-image">📦</div>'}
                </div>
                <div class="product-info">
                  <div class="product-name">
                    ${sanitize(it.product_name || it.name)}
                  </div>
                  <div class="product-details">
                    ${it.brand ? `<div class="product-detail-item"><span class="icon">🏷️</span>${sanitize(it.brand)}</div>` : ''}
                    ${it.variant ? `<div class="product-detail-item"><span class="icon">📦</span>${sanitize(it.variant)}</div>` : ''}
                    ${it.eta ? `<div class="product-detail-item"><span class="icon">🚚</span>${sanitize(it.eta)}</div>` : ''}
                    ${it.inventory > 0 ? `<div class="product-detail-item"><span class="icon">📦</span>Stock: ${it.inventory}</div>` : ''}
                  </div>
                  ${it.product_url ? `<a href="${it.product_url}" target="_blank" class="product-link">View Product</a>` : ''}
                </div>
                <div class="price-info">
                  <div class="total-price">₹ ${it.price.toLocaleString('en-IN')}</div>
                  ${it.rating && it.rating !== 'N/A' ? `<div class="rating-stars">${it.rating}</div>` : '<div class="muted">No rating</div>'}
                  ${it.review_count ? `<div class="review-count">${it.review_count} reviews</div>` : ''}
                </div>
              </div>
            `)
            .join('')}
        </div>
        <button class="btn" data-platform="${platform}">View on ${platform}</button>
      `;
      const button = card.querySelector('button');
      button.addEventListener('click', () => redirectToCart(platform));
      platformCards.appendChild(card);
    }
  }

  function redirectToCart(platform) {
    let url;
    if (platform.toLowerCase() === 'zepto') {
      url = 'https://www.zeptonow.com/';
    } else if (platform.toLowerCase() === 'bigbasket') {
      url = 'https://www.bigbasket.com/';
    } else if (platform.toLowerCase() === 'jiomart') {
      url = 'https://www.jiomart.com/';
    } else {
      const base = platform.toLowerCase();
      const params = new URLSearchParams();
      params.set('items', JSON.stringify(state.items));
      url = `https://www.${base}.in/search?` + params.toString();
    }
    
    showToast(`Opening ${platform}...`);
    window.open(url, '_blank');
  }

  // Single search handler
  async function handleSingleSearch() {
    const itemName = singleItemInput.value.trim();
    if (!itemName) {
      showToast('Please enter an item to search.');
      return;
    }

    try {
      showToast('Searching for prices...');
      searchSingleBtn.disabled = true;
      searchSingleBtn.innerHTML = '<span class="loading-spinner"></span>Searching...';
      
      // Create a single item for the search
      const singleItem = {
        item_name: itemName,
        quantity: '1',
        unit: 'pieces'
      };

      // Show loading cards
      renderLoadingCards();
      
      const result = await searchPricesAPI([singleItem]);
      
      if (result.success) {
        const priceData = processPriceData(result.results);
        renderCards(priceData);
        showRelatedItems(itemName);
        showToast(`Found prices for ${itemName}!`);
      } else {
        showToast('Failed to find prices. Please try again.');
        renderCards(null);
      }
    } catch (error) {
      console.error('Error searching prices:', error);
      showToast('Error searching prices. Please try again.');
      renderCards(null);
    } finally {
      searchSingleBtn.disabled = false;
      searchSingleBtn.innerHTML = '<span class="btn-text">Search Prices</span><span class="btn-icon">🔍</span>';
    }
  }

  function handleReset() {
    clearAllData();
    showToast('Cleared.');
  }

  // Event listeners
  resetBtn.addEventListener('click', handleReset);
  searchSingleBtn.addEventListener('click', handleSingleSearch);

  // Autocomplete event listeners
  singleItemInput.addEventListener('input', (e) => {
    const query = e.target.value;
    showAutocomplete(query);
    updateSearchButton();
  });

  singleItemInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      if (!searchSingleBtn.disabled) {
        handleSingleSearch();
      }
    } else if (e.key === 'Escape') {
      hideAutocomplete();
    }
  });

  // Hide autocomplete when clicking outside
  document.addEventListener('click', (e) => {
    if (!singleItemInput.contains(e.target) && !autocompleteDropdown.contains(e.target)) {
      hideAutocomplete();
    }
  });

  // Suggestion tag event listeners
  document.querySelectorAll('.suggestion-tag').forEach(tag => {
    tag.addEventListener('click', () => {
      const itemName = tag.dataset.item;
      singleItemInput.value = itemName;
      updateSearchButton();
      hideAutocomplete();
    });
  });

  // Initialize
  renderCards(null);
})();
