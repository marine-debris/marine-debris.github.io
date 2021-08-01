<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.18.2-Zürich" styleCategories="AllStyleCategories" maxScale="0" hasScaleBasedVisibilityFlag="0" minScale="1e+08">
  <flags>
    <Identifiable>1</Identifiable>
    <Removable>1</Removable>
    <Searchable>1</Searchable>
    <Private>0</Private>
  </flags>
  <temporal fetchMode="0" enabled="0" mode="0">
    <fixedRange>
      <start></start>
      <end></end>
    </fixedRange>
  </temporal>
  <customproperties>
    <property key="WMSBackgroundLayer" value="false"/>
    <property key="WMSPublishDataSourceUrl" value="false"/>
    <property key="embeddedWidgets/count" value="0"/>
    <property key="identify/format" value="Value"/>
  </customproperties>
  <pipe>
    <provider>
      <resampling enabled="false" zoomedInResamplingMethod="nearestNeighbour" zoomedOutResamplingMethod="nearestNeighbour" maxOversampling="2"/>
    </provider>
    <rasterrenderer band="1" opacity="1" alphaBand="-1" nodataColor="" type="paletted">
      <rasterTransparency/>
      <minMaxOrigin>
        <limits>None</limits>
        <extent>WholeRaster</extent>
        <statAccuracy>Estimated</statAccuracy>
        <cumulativeCutLower>0.02</cumulativeCutLower>
        <cumulativeCutUpper>0.98</cumulativeCutUpper>
        <stdDevFactor>2</stdDevFactor>
      </minMaxOrigin>
      <colorPalette>
        <paletteEntry alpha="255" label="Marine Debris" value="1" color="#ff0000"/>
        <paletteEntry alpha="255" label="Dense Sargassum" value="2" color="#008000"/>
        <paletteEntry alpha="255" label="Sparse Sargassum" value="3" color="#32cd32"/>
        <paletteEntry alpha="255" label="Natural Organic Material" value="4" color="#b22222"/>
        <paletteEntry alpha="255" label="Ship" value="5" color="#ffa500"/>
        <paletteEntry alpha="255" label="Clouds" value="6" color="#c0c0c0"/>
        <paletteEntry alpha="255" label="Marine Water" value="7" color="#000080"/>
        <paletteEntry alpha="255" label="Sediment-Laden Water" value="8" color="#ffd700"/>
        <paletteEntry alpha="255" label="Foam" value="9" color="#800080"/>
        <paletteEntry alpha="255" label="Turbid Water" value="10" color="#bdb76b"/>
        <paletteEntry alpha="255" label="Shallow Water" value="11" color="#00ced1"/>
        <paletteEntry alpha="255" label="Waves" value="12" color="#fff5ee"/>
        <paletteEntry alpha="255" label="Cloud Shadows" value="13" color="#808080"/>
        <paletteEntry alpha="255" label="Wakes" value="14" color="#ffff00"/>
        <paletteEntry alpha="255" label="Mixed Water" value="15" color="#bc8f8f"/>
      </colorPalette>
      <colorramp name="[source]" type="randomcolors">
        <Option/>
      </colorramp>
    </rasterrenderer>
    <brightnesscontrast brightness="0" contrast="0" gamma="1"/>
    <huesaturation colorizeOn="0" saturation="0" grayscaleMode="0" colorizeRed="255" colorizeGreen="128" colorizeBlue="128" colorizeStrength="100"/>
    <rasterresampler maxOversampling="2"/>
    <resamplingStage>resamplingFilter</resamplingStage>
  </pipe>
  <blendMode>0</blendMode>
</qgis>
