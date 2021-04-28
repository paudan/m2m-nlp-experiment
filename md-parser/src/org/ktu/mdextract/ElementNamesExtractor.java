package org.ktu.mdextract;

import com.nomagic.magicdraw.core.Application;
import com.nomagic.magicdraw.core.Project;
import com.nomagic.magicdraw.uml.symbols.DiagramPresentationElement;
import com.nomagic.uml2.ext.magicdraw.classes.mdkernel.Association;
import com.nomagic.uml2.ext.magicdraw.classes.mdkernel.Diagram;
import com.nomagic.uml2.ext.magicdraw.classes.mdkernel.DirectedRelationship;
import com.nomagic.uml2.ext.magicdraw.classes.mdkernel.Element;
import com.nomagic.uml2.ext.magicdraw.classes.mdkernel.Generalization;
import com.nomagic.uml2.ext.magicdraw.classes.mdkernel.NamedElement;
import com.nomagic.uml2.ext.magicdraw.classes.mdkernel.Package;
import com.nomagic.uml2.ext.magicdraw.classes.mdkernel.Type;
import com.nomagic.uml2.ext.magicdraw.mdusecases.Actor;
import com.nomagic.uml2.ext.magicdraw.mdusecases.Extend;
import com.nomagic.uml2.ext.magicdraw.mdusecases.ExtensionPoint;
import com.nomagic.uml2.ext.magicdraw.mdusecases.Include;
import com.nomagic.uml2.ext.magicdraw.mdusecases.UseCase;
import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Paths;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Objects;
import java.util.Set;
import java.util.function.Function;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.Collectors;
import java.util.stream.Stream;


public class ElementNamesExtractor {

    protected Collection<DiagramPresentationElement> diagrams;
    protected Collection<Element> candidateElements = new HashSet<>();
    protected Map<DiagramPresentationElement, Collection<Element>> elementsByDiagram = new HashMap<>();

    public ElementNamesExtractor(Package root) {
        this.diagrams = getDiagrams(root);
        for (DiagramPresentationElement diagram: diagrams)
            readElements(diagram);
    }

    public ElementNamesExtractor(Collection<DiagramPresentationElement> diagrams) {
        this.diagrams = diagrams;
        for (DiagramPresentationElement diagram: diagrams)
            readElements(diagram);
    }

    protected void readElements(DiagramPresentationElement diagram) {
        Collection<Element> diagramElements = diagram.getUsedModelElements();
        candidateElements.addAll(diagramElements);
        elementsByDiagram.put(diagram, diagramElements);
    }

    protected Collection<Element> getDiagramElements(Collection<Element> colelem, DiagramPresentationElement diagram) {
        Collection<Element> newelem = new HashSet<>();
        for (Element element : colelem)
            if (element instanceof Package && diagram.findPresentationElement(element, null) != null)
                addPackageElements(diagram, (Package) element, newelem);
        return newelem;
    }

    protected void addPackageElements(DiagramPresentationElement diagram, Package pack, Collection<Element> elements) {
        for (Element el : pack.getOwnedElement())
            if (diagram.findPresentationElement(el, null) != null)
                elements.add(el);
        for (Package innerPack : pack.getNestedPackage())
            if (diagram.findPresentationElement(innerPack, null) != null)
                addPackageElements(diagram, innerPack, elements);
    }

    protected Collection<DiagramPresentationElement> getDiagrams(Package model, Collection<DiagramPresentationElement> diagrams) {
        Project project = Application.getInstance().getProject();
        if (project == null || model == null)
            return diagrams;
        for (Diagram diag : model.getOwnedDiagram()) {
            DiagramPresentationElement pres = project.getDiagram(diag);
            if (pres != null)
                diagrams.add(pres);
        }
        for (Package pkg : model.getNestedPackage())
            diagrams = getDiagrams(pkg, diagrams);
        return diagrams;
    }

    public Collection<DiagramPresentationElement> getDiagrams(Package root) {
        Collection<DiagramPresentationElement> diagrams = new HashSet<>();
        return getDiagrams(root, diagrams);
    }


    public void extractDiagramsPerPackage(String prefix) {
        Map<String, String> data = elementsByDiagram.keySet().stream()
                .filter(e -> e.getDiagram().getOwner() instanceof Package)
                .collect(Collectors.toMap(DiagramPresentationElement::getName, e -> ((Package) e.getDiagram().getOwner()).getName()));
        prefix = prefix == null ? "" : (prefix + "_");
        String output = String.join("\n", data.entrySet().stream().map(p -> p.getKey() + ";" + p.getValue()).collect(Collectors.toSet()));
        writeFile(Paths.get("output", prefix + "diagrams.csv").toString(), output);
    }

    public void extract(String prefix) {
        prefix = prefix == null ? "" : (prefix + "_");
        Set<UseCase> useCases = candidateElements.stream().filter(e -> e instanceof UseCase).map(e -> (UseCase)e).collect(Collectors.toSet());
        String output = String.join("\n", useCases.stream().map(NamedElement::getName).collect(Collectors.toSet()));
        writeFile(Paths.get("output", prefix + "useCases.csv").toString(), output);
        output = String.join("\n", extractAssociations(candidateElements).stream().map(p -> String.join(";", p)).collect(Collectors.toSet()));
        writeFile(Paths.get("output", prefix + "actors_useCases.csv").toString(), output);
    }

    public void extractByDiagram(String prefix, String filename, Function<Entry<DiagramPresentationElement, Collection<Element>>, Set<String[]>> func) {
        prefix = prefix == null ? "" : (prefix + "_");
        if (filename == null)
            filename = "actors_useCases_diag";
        filename = Paths.get("output", prefix + filename + ".csv").toString();
        try (BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(filename), StandardCharsets.UTF_8))) {
            for (Entry<DiagramPresentationElement, Collection<Element>> entry: elementsByDiagram.entrySet()) {
                String diagramName = entry.getKey().getName();
                if (diagramName == null) continue;
                Set<String[]> entries = func.apply(entry);
                if (entries.size() == 0) continue;
                String output = String.join("\n", entries.stream().map(p -> diagramName + ";" + String.join(";", p)).collect(Collectors.toSet()));
                writer.write(output);
                writer.newLine();
            }
        } catch (IOException e) {
            Logger.getLogger(ElementNamesExtractor.class.getName()).log(Level.SEVERE, null, e);
        }
    }

    protected Set<String[]> extractByType(Collection<Element> candidateElements, Class<?> cl) {
        return candidateElements.stream()
                .filter(el -> el.getClassType().equals(cl))
                .filter(el -> ((NamedElement)el).getName().trim().length() > 0)
                .map(el -> new String[] {((NamedElement)el).getName()})
                .collect(Collectors.toSet());
    }

    public Set<String[]> extractUseCases(Collection<Element> candidateElements) {
        return extractByType(candidateElements, UseCase.class);
    }

    public Set<String[]> extractActors(Collection<Element> candidateElements) {
        return extractByType(candidateElements, Actor.class);
    }

    protected Set<String[]> extractActorGeneralizations(Collection<Element> candidateElements) {
        return candidateElements.stream()
                .filter(el -> el.getClassType().equals(Generalization.class))
                .map(el -> (Generalization)el)
                .filter(el -> Objects.requireNonNull(el.getSpecific()).getClassType().equals(Actor.class) &&
                              Objects.requireNonNull(el.getGeneral()).getClassType().equals(Actor.class))
                .map(el -> new String[] {el.getSpecific().getName(), el.getGeneral().getName()})
                .collect(Collectors.toSet());
    }

    public Set<String[]> extractAssociations(Collection<Element> candidateElements) {
        Set<Association> associations = candidateElements.stream()
                .filter(el -> el.getClassType().equals(Association.class))
                .filter(el -> {
                    Collection<Type> endtypes = ((Association) el).getEndType();
                    boolean actor_found = false, uc_found = false;
                    for (Type elem : endtypes)
                        if (elem.getClassType().equals(Actor.class))
                            actor_found = true;
                        else if (elem.getClassType().equals(UseCase.class))
                            uc_found = true;
                    return actor_found && uc_found;
                }).map(e -> (Association)e)
                .collect(Collectors.toSet());
        return associations.stream()
                .map(el -> {
                    Collection<Type> endtypes = el.getEndType();
                    String actorName = null, ucName = null;
                    for (Type elem : endtypes)
                        if (elem.getClassType().equals(Actor.class))
                            actorName = elem.getName();
                        else if (elem.getClassType().equals(UseCase.class))
                            ucName = elem.getName();
                    return new String[] {actorName, ucName};
                })
                .filter(el -> el[0].trim().length() > 0 && el[1].trim().length() > 0)
                .collect(Collectors.toSet());
    }

    public Set<String[]> extractBoundaryAssociations(Collection<Element> candidateElements) {
        return candidateElements.stream()
                .filter(el -> el.getClassType().equals(UseCase.class) && el.getOwner() instanceof Package)
                .map(e -> (UseCase)e)
                .filter(e -> e.get_associationOfEndType().size() == 0)
                .map(el -> new String[] {el.getName(), ((Package) Objects.requireNonNull(el.getOwner())).getName()})
                .filter(el -> el[0].trim().length() > 0 && el[1].trim().length() > 0)
                .collect(Collectors.toSet());
    }

    private Collection<Actor> addActors(Collection<Actor> actors, UseCase including) {
        if (actors == null)
            actors = new HashSet<>();
        for (Association ai : including.get_associationOfEndType())
            for (Type elem : ai.getEndType())
                if (elem.getClassType().equals(Actor.class) && candidateElements.contains(elem))
                    actors.add((Actor) elem);
        return actors;
    }

    private Collection<Actor> getActorsOfUseCase(UseCase including, DirectedRelationship exclude) {
        Collection<Actor> actors = new HashSet<>();
        Collection<DirectedRelationship> visited = new HashSet<>();
        actors = addActors(actors, including);
        Collection<DirectedRelationship> includes = new HashSet<>();
        for (Include include : including.get_includeOfAddition())
            if (candidateElements.contains(include))
                includes.add(include);
        for (Extend extend : including.get_extendOfExtendedCase())
            if (candidateElements.contains(extend))
                includes.add(extend);
        if (exclude != null)
            includes.remove(exclude);
        while (!includes.isEmpty()) {
            Collection<DirectedRelationship> newIncludes = new HashSet<>();
            for (DirectedRelationship include : includes) {
                UseCase uc = null;
                if (include instanceof Include)
                    uc = ((Include) include).getIncludingCase();
                else if (include instanceof Extend)
                    uc = ((Extend) include).getExtendedCase();
                if (uc != null) {
                    for (Include incl : uc.get_includeOfAddition())
                        // Exclude previously visited relationships
                        if (candidateElements.contains(incl) && !visited.contains(incl))
                            newIncludes.add(incl);
                    for (Extend extend : uc.get_extendOfExtendedCase())
                        if (candidateElements.contains(extend) && !visited.contains(extend))
                            newIncludes.add(extend);
                    actors = addActors(actors, uc);
                    // The first actors which are found is set to be executing the given UseCase
                    if (!actors.isEmpty())
                        return actors;
                }
            }
            visited.addAll(includes);
            includes = newIncludes;
        }
        return actors;
    }

    public Set<String[]> extractExtensionPoints(Collection<Element> candidateElements) {
        return candidateElements.stream()
                .filter(el -> el.getClassType().equals(Extend.class))
                .map(e -> (Extend)e)
                .flatMap(el -> el.getExtensionLocation().stream().map(ExtensionPoint::getName))
                .filter(el -> el.trim().length() > 0)
                .map(el -> new String[] {el})
                .collect(Collectors.toSet());
    }

    public Set<String[]> extractIncludeTuples(Collection<Element> candidateElements) {
        return candidateElements.stream()
                .filter(el -> el.getClassType().equals(Include.class))
                .flatMap(el -> {
                    Set<String[]> results = new HashSet<>();
                    UseCase including = ((Include) el).getIncludingCase();
                    UseCase included = ((Include) el).getAddition();
                    if (including == null || included == null)
                        return null;
                    if (including.getName().trim().length() == 0 || included.getName().trim().length() == 0)
                        return null;
                    // Must exclude current Include in order not to traverse it
                    Collection<Actor> aincluding = getActorsOfUseCase(including, (Include) el);
                    Collection<Actor> aincluded = getActorsOfUseCase(included, (Include) el);
                    if (aincluding.size() > 0 && aincluded.size() > 0)
                        for (Actor ai : aincluding)
                            for (Actor aai : aincluded)
                                results.add(new String[] {ai.getName(), including.getName(), aai.getName(), included.getName()});
                    else if (aincluding.isEmpty() && aincluded.size() > 0)
                        for (Actor aai : aincluded)
                            if (including.getOwner() instanceof NamedElement)
                                results.add(new String[] {((NamedElement)including.getOwner()).getName(), including.getName(), aai.getName(), included.getName()});
                    else if (aincluded.isEmpty() && aincluding.size() > 0)
                        for (Actor ai : aincluding)
                            results.add(new String[] {ai.getName(), including.getName(), ai.getName(), included.getName()});
                    else
                        if (including.getOwner() instanceof NamedElement && included.getOwner() instanceof NamedElement)
                            results.add(new String[] {((NamedElement)including.getOwner()).getName(), including.getName(), ((NamedElement)included.getOwner()).getName(), included.getName()});
                    return results.stream();
                }).collect(Collectors.toSet());
    }

    public Set<String[]> extractExtendTuples(Collection<Element> candidateElements) {
        return candidateElements.stream()
                .filter(el -> el.getClassType().equals(Extend.class))
                .flatMap(el -> {
                    Set<String[]> results = new HashSet<>();
                    UseCase extended = ((Extend) el).getExtendedCase();
                    UseCase extension = ((Extend) el).getExtension();
                    if (extended == null || extension == null)
                        return null;
                    if (extended.getName().trim().length() == 0 || extension.getName().trim().length() == 0)
                        return null;
                    Collection<Actor> aextended = getActorsOfUseCase(extended, (Extend) el);
                    Collection<Actor> aextension = getActorsOfUseCase(extension, (Extend) el);
                    if (aextended.size() > 0 && aextension.size() > 0)
                        for (Actor ai : aextended)
                            for (Actor aai : aextension)
                                results.add(new String[] {ai.getName(), extended.getName(), aai.getName(), extension.getName()});
                    else if (aextended.isEmpty() && aextension.size() > 0)
                        for (Actor aai : aextension)
                            if (extended.getOwner() instanceof NamedElement)
                                results.add(new String[] {((NamedElement)extended.getOwner()).getName(), extended.getName(), aai.getName(), extension.getName()});
                    else if (aextended.isEmpty() && aextension.isEmpty())
                        if (extended.getOwner() instanceof NamedElement && extension.getOwner() instanceof NamedElement)
                            results.add(new String[] {((NamedElement)extended.getOwner()).getName(), extended.getName(), ((NamedElement)extension.getOwner()).getName(), extension.getName()});
                    return results.stream();
                }).collect(Collectors.toSet());
    }

    public Set<String[]> extractUseCaseGeneralizationTuples(Collection<Element> candidateElements) {
        return candidateElements.stream()
                .filter(el -> el.getClassType().equals(Generalization.class))
                .map(e -> (Generalization)e)
                .filter(e -> Objects.requireNonNull(e.getGeneral()).getClassType().equals(UseCase.class) &&
                             Objects.requireNonNull(e.getSpecific()).getClassType().equals(UseCase.class))
                .flatMap(e -> e.getGeneral().get_associationOfEndType().stream()
                            .flatMap(ae -> ae.getEndType().stream().filter(e2 -> e2.getClassType().equals(Actor.class)))
                            .map(ae -> new String[]{ae.getName(), e.getSpecific().getName()})
                ).collect(Collectors.toSet());
    }

    public Set<String[]> extractUseCaseGeneralizations(Collection<Element> candidateElements) {
        return candidateElements.stream()
                .filter(el -> el.getClassType().equals(Generalization.class))
                .map(e -> (Generalization)e)
                .filter(e -> Objects.requireNonNull(e.getGeneral()).getClassType().equals(UseCase.class) &&
                        Objects.requireNonNull(e.getSpecific()).getClassType().equals(UseCase.class))
                .map(el -> new String[] {el.getGeneral().getName(), el.getSpecific().getName()})
                .collect(Collectors.toSet());
    }

    public Set<String[]> extractIncludeRelations(Collection<Element> candidateElements) {
        return candidateElements.stream()
                .filter(el -> el.getClassType().equals(Include.class))
                .map(e -> (Include)e)
                .map(el -> new String[] {Objects.requireNonNull(el.getIncludingCase()).getName(), Objects.requireNonNull(el.getAddition()).getName()})
                .collect(Collectors.toSet());
    }

    public Set<String[]> extractExtendRelations(Collection<Element> candidateElements) {
        return candidateElements.stream()
                .filter(el -> el.getClassType().equals(Extend.class))
                .map(e -> (Extend)e)
                .map(el -> new String[] {Objects.requireNonNull(el.getExtendedCase()).getName(), Objects.requireNonNull(el.getExtension()).getName()})
                .collect(Collectors.toSet());
    }

    public void extractByDiagram(String prefix) {
        Function<Entry<DiagramPresentationElement, Collection<Element>>, Set<String[]>> func = entry -> extractAssociations(entry.getValue());
        this.extractByDiagram(prefix, null, func);
    }

    public void writeFile(String filename, String output) {
        try (BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(filename), StandardCharsets.UTF_8))) {
            writer.write(output);
        } catch (IOException e) {
            Logger.getLogger(ElementNamesExtractor.class.getName()).log(Level.SEVERE, null, e);
        }
    }
}

