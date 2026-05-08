---
"@oxide-js/layers": patch
"@oxide-js/models": patch
"@oxide-js/core": patch
---

fix: point package exports to compiled JS files to avoid type-stripping errors in Node.js v25+
