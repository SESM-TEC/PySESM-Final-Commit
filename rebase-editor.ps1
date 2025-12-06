param($todoFile)
$content = Get-Content $todoFile -Raw
# Reemplazar 'pick' con 'squash' en todas excepto la última línea
$lines = $content -split "
"
for ($i = 0; $i -lt $lines.Count - 1; $i++) {
    if ($lines[$i] -match '^pick') {
        $lines[$i] = $lines[$i] -replace '^pick', 'squash'
    }
}
$lines -join "
" | Set-Content $todoFile
