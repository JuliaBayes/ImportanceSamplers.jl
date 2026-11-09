```@eval
import Markdown
Markdown.parse(read(joinpath(@__DIR__, "..", "..", "CHANGELOG.md"), String))
```
