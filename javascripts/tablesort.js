// Source: https://squidfunk.github.io/mkdocs-material/reference/data-tables/?h=table#sortable-tables-docsjavascriptstablesortjs
document$.subscribe(function() {
    var tables = document.querySelectorAll("article table:not([class])")
    tables.forEach(function(table) {
      new Tablesort(table)
    })
  })