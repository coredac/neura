#include "NeuraDialect/Util/ArchParser.h"
#include "NeuraDialect/Util/NeuraYamlKeys.h"
#include "NeuraDialect/Util/ParserUtils.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir::neura::yamlkeys;

namespace mlir {
namespace neura {
namespace util {

ArchParser::ArchParser(const std::string &architecture_spec_file)
    : architecture_spec_file(architecture_spec_file) {}

mlir::FailureOr<Architecture> ArchParser::getArchitecture() {
  // Default values for architecture specification file.
  // By default Neura assumes a single CGRA unless a multi-CGRA architecture
  // specification is provided explicitly.
  constexpr int kMultiCgraDefaultRows = 1;
  constexpr int kMultiCgraDefaultColumns = 1;
  constexpr int kPerCgraDefaultRows = 4;
  constexpr int kPerCgraDefaultColumns = 4;
  constexpr int kDefaultMaxCtrlMemItems = 20;

  int multi_cgra_rows = kMultiCgraDefaultRows;
  int multi_cgra_columns = kMultiCgraDefaultColumns;
  int per_cgra_rows = kPerCgraDefaultRows;
  int per_cgra_columns = kPerCgraDefaultColumns;
  int max_ctrl_mem_items = kDefaultMaxCtrlMemItems;
  mlir::neura::TileDefaults tile_defaults;
  std::vector<mlir::neura::TileOverride> tile_overrides;
  mlir::neura::LinkDefaults link_defaults;
  std::vector<mlir::neura::LinkOverride> link_overrides;
  mlir::neura::BaseTopology multi_cgra_base_topology =
      mlir::neura::BaseTopology::MESH;
  mlir::neura::BaseTopology per_cgra_base_topology =
      mlir::neura::BaseTopology::MESH;

  if (!architecture_spec_file.empty()) {

    // Use LLVM YAML parser to validate the YAML syntax (no mapping yet)
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer_or_err =
        llvm::MemoryBuffer::getFile(architecture_spec_file);
    if (!buffer_or_err) {
      llvm::errs() << "Failed to open architecture "
                      "specification file: "
                   << architecture_spec_file << "\n";
      return failure();
    }

    llvm::SourceMgr sm;
    sm.AddNewSourceBuffer(std::move(*buffer_or_err), llvm::SMLoc());
    llvm::yaml::Stream yaml_stream(
        sm.getMemoryBuffer(sm.getMainFileID())->getBuffer(), sm);

    bool parse_failed = false;
    llvm::yaml::Document &yaml_doc = *yaml_stream.begin();
    (void)yaml_doc; // ensure document is created
    if (yaml_stream.failed()) {
      parse_failed = true;
    }

    if (parse_failed) {
      llvm::errs() << "YAML parse error in: " << architecture_spec_file << "\n";
      return failure();
    }

    // Parse YAML configuration
    if (!parseArchitectureYaml(yaml_doc, multi_cgra_rows, multi_cgra_columns,
                               multi_cgra_base_topology, per_cgra_rows,
                               per_cgra_columns, per_cgra_base_topology,
                               max_ctrl_mem_items, tile_defaults,
                               tile_overrides, link_defaults, link_overrides)) {
      return failure();
    }
  } else {
    llvm::errs() << "No architecture specification "
                    "file provided.\n";
  }

  return Architecture(multi_cgra_rows, multi_cgra_columns,
                      multi_cgra_base_topology, per_cgra_rows, per_cgra_columns,
                      max_ctrl_mem_items, per_cgra_base_topology, tile_defaults,
                      tile_overrides, link_defaults, link_overrides);
}

bool ArchParser::parseArchitectureYaml(
    llvm::yaml::Document &doc, int &multi_cgra_rows, int &multi_cgra_columns,
    mlir::neura::BaseTopology &multi_cgra_base_topology, int &per_cgra_rows,
    int &per_cgra_columns, mlir::neura::BaseTopology &per_cgra_base_topology,
    int &max_ctrl_mem_items, mlir::neura::TileDefaults &tile_defaults,
    std::vector<mlir::neura::TileOverride> &tile_overrides,
    mlir::neura::LinkDefaults &link_defaults,
    std::vector<mlir::neura::LinkOverride> &link_overrides) {
  auto *root = doc.getRoot();
  if (!root)
    return yamlParseError("Empty YAML document");
  auto *root_map = llvm::dyn_cast<llvm::yaml::MappingNode>(root);
  if (!root_map)
    return yamlParseError("YAML root is not a mapping");

  for (auto &key_value_pair : *root_map) {
    auto *key_node =
        llvm::dyn_cast_or_null<llvm::yaml::ScalarNode>(key_value_pair.getKey());
    if (!key_node)
      continue;
    llvm::SmallString<64> key_string;
    llvm::StringRef key_ref = key_node->getValue(key_string);

    if (key_ref == kArchitecture) {
      // Not used in this parser, but could be handled here.
      continue;
    } else if (key_ref == kMultiCgraDefaults) {
      auto *multi_cgra_map = llvm::dyn_cast_or_null<llvm::yaml::MappingNode>(
          key_value_pair.getValue());
      if (!multi_cgra_map)
        continue;
      for (auto &multi_cgra_map_key_value_pair : *multi_cgra_map) {
        auto *multi_cgra_map_key_node =
            llvm::dyn_cast_or_null<llvm::yaml::ScalarNode>(
                multi_cgra_map_key_value_pair.getKey());
        if (!multi_cgra_map_key_node)
          continue;
        llvm::SmallString<64> multi_cgra_map_key_string;
        llvm::StringRef multi_cgra_map_key_ref =
            multi_cgra_map_key_node->getValue(multi_cgra_map_key_string);
        int temp_value = 0;
        if (multi_cgra_map_key_ref == kRows) {
          if (parseYamlScalarInt(multi_cgra_map_key_value_pair.getValue(),
                                 temp_value))
            multi_cgra_rows = temp_value;
        } else if (multi_cgra_map_key_ref == kColumns) {
          if (parseYamlScalarInt(multi_cgra_map_key_value_pair.getValue(),
                                 temp_value))
            multi_cgra_columns = temp_value;
        } else if (multi_cgra_map_key_ref == kBaseTopology) {
          std::string topo_str;
          if (parseYamlScalarString(multi_cgra_map_key_value_pair.getValue(),
                                    topo_str))
            multi_cgra_base_topology = parseTopologyString(topo_str);
        }
      }
    } else if (key_ref == kPerCgraDefaults) {
      auto *per_cgra_map = llvm::dyn_cast_or_null<llvm::yaml::MappingNode>(
          key_value_pair.getValue());
      if (!per_cgra_map)
        continue;
      for (auto &per_cgra_map_key_value_pair : *per_cgra_map) {
        auto *per_cgra_map_key_node =
            llvm::dyn_cast_or_null<llvm::yaml::ScalarNode>(
                per_cgra_map_key_value_pair.getKey());
        if (!per_cgra_map_key_node)
          continue;
        llvm::SmallString<64> per_cgra_map_key_string;
        llvm::StringRef per_cgra_map_key_ref =
            per_cgra_map_key_node->getValue(per_cgra_map_key_string);
        int temp_value = 0;
        if (per_cgra_map_key_ref == kRows) {
          if (parseYamlScalarInt(per_cgra_map_key_value_pair.getValue(),
                                 temp_value))
            per_cgra_rows = temp_value;
        } else if (per_cgra_map_key_ref == kColumns) {
          if (parseYamlScalarInt(per_cgra_map_key_value_pair.getValue(),
                                 temp_value))
            per_cgra_columns = temp_value;
        } else if (per_cgra_map_key_ref == kBaseTopology) {
          std::string topo_str;
          if (parseYamlScalarString(per_cgra_map_key_value_pair.getValue(),
                                    topo_str))
            per_cgra_base_topology = parseTopologyString(topo_str);
        } else if (per_cgra_map_key_ref == kCtrlMemItems) {
          if (parseYamlScalarInt(per_cgra_map_key_value_pair.getValue(),
                                 temp_value))
            max_ctrl_mem_items = temp_value;
        }
      }
    } else if (key_ref == kTileDefaults) {
      auto *tile_defaults_map = llvm::dyn_cast_or_null<llvm::yaml::MappingNode>(
          key_value_pair.getValue());
      if (tile_defaults_map)
        parseTileDefaults(tile_defaults_map, tile_defaults);
    } else if (key_ref == kTileOverrides) {
      auto *tile_overrides_seq =
          llvm::dyn_cast_or_null<llvm::yaml::SequenceNode>(
              key_value_pair.getValue());
      if (tile_overrides_seq)
        parseTileOverrides(tile_overrides_seq, tile_overrides);
    } else if (key_ref == kLinkDefaults) {
      auto *link_defaults_map = llvm::dyn_cast_or_null<llvm::yaml::MappingNode>(
          key_value_pair.getValue());
      if (link_defaults_map)
        parseLinkDefaults(link_defaults_map, link_defaults);
    } else if (key_ref == kLinkOverrides) {
      auto *link_overrides_seq =
          llvm::dyn_cast_or_null<llvm::yaml::SequenceNode>(
              key_value_pair.getValue());
      if (link_overrides_seq)
        parseLinkOverrides(link_overrides_seq, link_overrides);
    } else {
      llvm::errs() << "Unknown YAML root key: " << key_ref << "\n";
    }
  }
  return true;
}
} // namespace util
} // namespace neura
} // namespace mlir
