document.addEventListener('DOMContentLoaded', function() {
    const sidebar = document.getElementById('sidebar');
    const content = document.getElementById('content');
    const toggleBtn = document.querySelector('.sidebar-toggle');

    // Загружаем состояние сайдбара из localStorage
    const sidebarState = localStorage.getItem('sidebarState');
    if (sidebarState === 'collapsed') {
        sidebar.classList.add('collapsed');
        content.classList.add('expanded');
    }

    toggleBtn.addEventListener('click', function() {
        sidebar.classList.toggle('collapsed');
        content.classList.toggle('expanded');
        
        // Сохраняем состояние сайдбара
        localStorage.setItem('sidebarState', 
            sidebar.classList.contains('collapsed') ? 'collapsed' : 'expanded'
        );
    });

    // Добавляем активный класс для текущей страницы
    const currentPath = window.location.pathname;
    const links = document.querySelectorAll('#sidebar ul li a');
    
    links.forEach(link => {
        if (link.getAttribute('href') === currentPath) {
            link.parentElement.classList.add('active');
        }
    });
});

// Сохраняем оригинальный код для загрузки файла
document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('upload-form');
    if (uploadForm) {
        uploadForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const formData = new FormData();
            const fileInput = document.getElementById('file-input');
            formData.append('file', fileInput.files[0]);
            
            try {
                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                if (response.ok) {
                    document.getElementById('status').textContent = result.message;
                    document.getElementById('table-container').innerHTML = result.table_html;
                } else {
                    document.getElementById('status').textContent = result.error;
                }
            } catch (error) {
                document.getElementById('status').textContent = 'Error uploading file';
            }
        });
    }
});

document.addEventListener('DOMContentLoaded', function() {
    // Подсветка активной вкладки
    const currentPath = window.location.pathname;
    const navLinks = document.querySelectorAll('.nav-links li a');
    
    navLinks.forEach(link => {
        if (link.getAttribute('href') === currentPath) {
            link.classList.add('active');
        }
    });
});

document.addEventListener('DOMContentLoaded', function() {
    const themeToggle = document.getElementById('theme-toggle');
    const body = document.body;

    themeToggle.addEventListener('click', function() {
        body.classList.toggle('dark-theme');
        localStorage.setItem('theme', body.classList.contains('dark-theme') ? 'dark' : 'light');
    });

    // Load theme from local storage
    if (localStorage.getItem('theme') === 'dark') {
        body.classList.add('dark-theme');
    }
});