$ErrorActionPreference = "Stop"

$projectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$batPath = Join-Path $projectDir "start_minion_chat.bat"
$desktopAppPath = Join-Path $projectDir "start_minion_desktop.ps1"
$assetDir = Join-Path $projectDir "assets"
$iconPath = Join-Path $assetDir "minion-chat.ico"

if (-not (Test-Path $desktopAppPath)) {
    throw "Khong tim thay file launcher: $desktopAppPath"
}

if (-not (Test-Path $assetDir)) {
    New-Item -ItemType Directory -Path $assetDir | Out-Null
}

function New-MinionChatIcon {
    param([string]$Path)

    Add-Type -AssemblyName System.Drawing

    function New-RoundedRectPath {
        param(
            [float]$X,
            [float]$Y,
            [float]$Width,
            [float]$Height,
            [float]$Radius
        )

        $path = New-Object System.Drawing.Drawing2D.GraphicsPath
        $diameter = $Radius * 2
        $path.AddArc($X, $Y, $diameter, $diameter, 180, 90)
        $path.AddArc($X + $Width - $diameter, $Y, $diameter, $diameter, 270, 90)
        $path.AddArc($X + $Width - $diameter, $Y + $Height - $diameter, $diameter, $diameter, 0, 90)
        $path.AddArc($X, $Y + $Height - $diameter, $diameter, $diameter, 90, 90)
        $path.CloseFigure()
        return $path
    }

    $bitmap = New-Object System.Drawing.Bitmap 256, 256
    $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
    $graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
    $graphics.Clear([System.Drawing.Color]::Transparent)

    $yellow = [System.Drawing.Brushes]::Gold
    $denim = New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::FromArgb(47, 111, 179))
    $dark = New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::FromArgb(36, 41, 51))
    $metal = New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::FromArgb(207, 215, 223))
    $glass = New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::FromArgb(255, 247, 214))
    $blackPen = New-Object System.Drawing.Pen ([System.Drawing.Color]::FromArgb(36, 41, 51)), 12
    $blackPen.StartCap = [System.Drawing.Drawing2D.LineCap]::Round
    $blackPen.EndCap = [System.Drawing.Drawing2D.LineCap]::Round

    # Icon mascot goc: mau vang, deo kinh bao ho, khong sao chep nhan vat ban quyen.
    $bodyPath = New-RoundedRectPath -X 36 -Y 8 -Width 184 -Height 240 -Radius 76
    $pantsPath = New-RoundedRectPath -X 36 -Y 150 -Width 184 -Height 98 -Radius 26
    $strapPath = New-RoundedRectPath -X 24 -Y 86 -Width 208 -Height 34 -Radius 16

    $graphics.FillPath($yellow, $bodyPath)
    $graphics.FillPath($denim, $pantsPath)
    $graphics.FillPath($dark, $strapPath)
    $graphics.FillEllipse($metal, 72, 54, 112, 112)
    $graphics.FillEllipse($glass, 88, 70, 80, 80)
    $graphics.FillEllipse($dark, 116, 98, 24, 24)
    $graphics.DrawArc($blackPen, 90, 150, 76, 52, 25, 130)

    $hicon = $bitmap.GetHicon()
    try {
        $icon = [System.Drawing.Icon]::FromHandle($hicon)
        $stream = [System.IO.File]::Open($Path, [System.IO.FileMode]::Create)
        try { $icon.Save($stream) } finally { $stream.Close(); $icon.Dispose() }
    } finally {
        $graphics.Dispose()
        $bitmap.Dispose()
        $bodyPath.Dispose()
        $pantsPath.Dispose()
        $strapPath.Dispose()
    }
}

if (-not (Test-Path $iconPath)) {
    New-MinionChatIcon -Path $iconPath
}

$desktopCandidates = @(
    [Environment]::GetFolderPath("Desktop"),
    (Join-Path $env:USERPROFILE "Desktop"),
    (Join-Path $env:USERPROFILE "OneDrive\Desktop"),
    [Environment]::GetFolderPath("CommonDesktopDirectory")
) | Where-Object { $_ -and (Test-Path $_) } | Select-Object -Unique

$shell = New-Object -ComObject WScript.Shell
foreach ($desktop in $desktopCandidates) {
    $shortcutPath = Join-Path $desktop "Minion Chat Local.lnk"
    $desktopLauncherPath = Join-Path $desktop "MinionChatLocal-launch.ps1"
    $desktopIconPath = Join-Path $desktop "MinionChatLocal.ico"
    try {
        Copy-Item -LiteralPath $iconPath -Destination $desktopIconPath -Force
        @(
            '$ErrorActionPreference = "Stop"'
            '$projectDir = "' + $projectDir.Replace('"', '""') + '"'
            '$desktopAppPath = Join-Path $projectDir "start_minion_desktop.ps1"'
            '$ps = Join-Path $env:SystemRoot "System32\WindowsPowerShell\v1.0\powershell.exe"'
            'Start-Process -FilePath $ps -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $desktopAppPath) -WindowStyle Hidden'
        ) | Set-Content -LiteralPath $desktopLauncherPath -Encoding UTF8

        $shortcut = $shell.CreateShortcut($shortcutPath)
        $shortcut.TargetPath = "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe"
        $shortcut.Arguments = "-NoProfile -ExecutionPolicy Bypass -File `"$desktopLauncherPath`""
        $shortcut.WorkingDirectory = $desktop
        $shortcut.Description = "Mo giao dien Minion Chat local"
        $shortcut.IconLocation = "$desktopIconPath,0"
        $shortcut.Save()
        Write-Host "Da tao shortcut: $shortcutPath"
    } catch {
        Write-Host "Bo qua vi tri khong co quyen ghi: $shortcutPath"
    }
}
