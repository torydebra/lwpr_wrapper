# LWPR_Wrapper

## Dependencies
- lwpr fork at https://github.com/torydebra/lwpr
- ros2 (some message utilities)

## How to compile
- `colcon build`, make sure you provide lwpr correct installation path

## How to use
### CmakeLists.txt
As any other ros2/colcon packages:
```cmake
find_package(lwpr_wrapper REQUIRED)
...
ament_target_dependencies(...
  ...
  lwpr_wrapper
)
```

### package.xml
```xml
  <depend>lwpr_wrapper</depend>
```

### Headers
```cpp
#include <lwpr_wrapper/LWPRWrapper.hpp>
...
lwpr_wrapper::LWPRWrapper lwpr_wrapper;
```

### Other
- `export LD_LIBRARY_PATH=/path/to/lwpr/lib:$LD_LIBRARY_PATH`