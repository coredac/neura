#ifndef NEURA_TEMPLATE_MAPPING_H
#define NEURA_TEMPLATE_MAPPING_H

#include "NeuraDialect/Mapping/Mapping.h"
#include "NeuraDialect/NeuraAttributes.h"

#include <set>

namespace mlir {
namespace neura {

class TemplateMapping : public Mapping {
public:
  bool map(std::vector<std::pair<Operation *, int>> &sorted_ops_with_levels,
           std::set<Operation *> &critical_ops,
           const Architecture &architecture,
           MappingState &mapping_state) override;

  std::string getName() const override { return attr::val::kTemplate.str(); }
};

} // namespace neura
} // namespace mlir

#endif // NEURA_TEMPLATE_MAPPING_H