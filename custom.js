(function () {
   'use strict';
    require(['base/js/namespace', 'jquery', 'base/js/events'], function(IPython, $, events){
        events.on('app_initialized.NotebookApp', function(){
            $('#header-container').hide();
            $('.header-bar').hide();
            $('div#maintoolbar').hide();
            IPython.menubar._size_header();
        });
        console.log('Header and toolbar should be hidden by default now');
    });
}());
